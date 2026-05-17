class LiveEEGReader:
    """Background reader for ThinkGear/NeuroSky USB serial EEG packets."""

    def __init__(self, port: Optional[str], baud: int):
        self.port = port
        self.baud = baud
        self.latest: Optional[dict] = None
        self.error: Optional[str] = EEG_IMPORT_ERROR
        self.packet_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if EEG_IMPORT_ERROR:
            self.error = EEG_IMPORT_ERROR
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop.clear()
            self.error = None
            self._thread = threading.Thread(target=self._run, name="eeg-live-reader", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            return dict(self.latest) if self.latest else None

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.error = message

    def _set_latest(self, reading: dict) -> None:
        with self._lock:
            self.latest = reading
            self.error = None
            self.packet_count += 1

    def _run(self) -> None:
        port = self.port or find_default_port()
        if not port:
            self._set_error("No USB serial EEG device found")
            return
        self.port = port

        try:
            with serial.Serial(port, self.baud, timeout=1.0) as stream:
                while not self._stop.is_set():
                    payload = read_packet(stream)
                    if payload is None:
                        continue

                    parsed = parse_payload(payload)
                    if not parsed or "eeg_power" not in parsed:
                        continue

                    bands = parsed["eeg_power"]
                    normalized = normalize_bands(bands)
                    reading = {
                        "online": True,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "port": port,
                        "baud": self.baud,
                        "poor_signal": parsed.get("poor_signal"),
                        "attention": parsed.get("attention"),
                        "meditation": parsed.get("meditation"),
                        "blink_strength": parsed.get("blink_strength"),
                        "eeg_power": bands,
                        "normalized_eeg": normalized,
                    }
                    for band in BAND_NAMES:
                        reading[band] = normalized.get(band, 0.0)

                    self._set_latest(reading)
        except Exception as exc:
            self._set_error(str(exc))


class LiveArduinoReader:
    """Background reader for PPG/GSR values emitted by the Arduino serial port."""

    def __init__(self, port: Optional[str], baud: int):
        self.port = port
        self.baud = baud
        self.latest: Optional[dict] = None
        self.error: Optional[str] = None
        self.packet_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if serial is None:
            self.error = "pyserial not available"
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop.clear()
            self.error = None
            self._thread = threading.Thread(target=self._run, name="arduino-live-reader", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            return dict(self.latest) if self.latest else None

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.error = message

    def _set_latest(self, reading: dict) -> None:
        with self._lock:
            self.latest = reading
            self.error = None
            self.packet_count += 1

    def _run(self) -> None:
        port = self.port or find_arduino_port()
        if not port:
            self._set_error("No Arduino serial device found")
            return
        self.port = port

        try:
            with serial.Serial(port, self.baud, timeout=1.0) as stream:
                time.sleep(1.5)
                while not self._stop.is_set():
                    raw = stream.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace")
                    parsed = parse_arduino_line(line)
                    if parsed:
                        parsed["port"] = port
                        parsed["baud"] = self.baud
                        self._set_latest(parsed)
        except Exception as exc:
            self._set_error(str(exc))


eeg_reader = LiveEEGReader(EEG_PORT, EEG_BAUD)
