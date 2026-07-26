import { useEffect } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { FacialEmotion, ShortcutMode } from '../types';

export function useSensorsStream() {
  const { 
    setShortcutMode, 
    setActiveEmotion, 
    tickJitter, 
    updateFromBackend, 
    setBackendConnected, 
    isLivePolling
  } = useDashboardStore();

  // Keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement as HTMLElement | null;
      if (
        activeEl && 
        (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.isContentEditable)
      ) {
        return;
      }

      const key = e.key.toUpperCase();

      // Facial shortcuts
      const facialMap: Record<string, FacialEmotion> = {
        'H': 'Happy',
        'S': 'Sad',
        'A': 'Angry',
        'N': 'Neutral',
        'F': 'Fear',
        'W': 'Surprise',
        'D': 'Disgust',
      };

      if (facialMap[key]) {
        e.preventDefault();
        setActiveEmotion(facialMap[key]);
        return;
      }

      // Mental state shortcuts (0 to 5)
      if (['0', '1', '2', '3', '4', '5'].includes(key)) {
        e.preventDefault();
        const mode = parseInt(key, 10) as ShortcutMode;
        setShortcutMode(mode);
        return;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setShortcutMode, setActiveEmotion]);

  // Periodic organic jitter tick
  useEffect(() => {
    const interval = setInterval(() => {
      tickJitter();
    }, 1600);

    return () => clearInterval(interval);
  }, [tickJitter]);

  // Backend live API polling
  useEffect(() => {
    if (!isLivePolling) return;

    const fetchBackend = async () => {
      try {
        const res = await fetch('/api/live_data?user_id=local_user');
        if (res.ok) {
          const data = await res.json();
          updateFromBackend(data);
        } else {
          setBackendConnected(false);
        }
      } catch (err) {
        setBackendConnected(false);
      }
    };

    fetchBackend();
    const interval = setInterval(fetchBackend, 3500);

    return () => clearInterval(interval);
  }, [isLivePolling, updateFromBackend, setBackendConnected]);
}
