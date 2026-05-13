import { useState, useRef, useEffect, useCallback } from 'react';

export function useUndoHistory(initialValue: string = '') {
  const [value, setValueState] = useState(initialValue);
  const historyRef = useRef<string[]>([initialValue]);
  const indexRef = useRef<number>(0);

  const setValue = useCallback((newValue: string | ((prev: string) => string)) => {
    setValueState((prev) => {
      const resolvedValue = typeof newValue === 'function' ? newValue(prev) : newValue;
      
      // We don't push history on every tiny change, we'll do it with a debounce.
      // But we always update the state
      return resolvedValue;
    });
  }, []);

  const reset = useCallback((newInitial: string = '') => {
    setValueState(newInitial);
    historyRef.current = [newInitial];
    indexRef.current = 0;
  }, []);

  // Use a debounced effect to save history
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      const currentHistoryValue = historyRef.current[indexRef.current];
      if (currentHistoryValue !== value) {
        // truncate redo history if typing again
        const newHistory = historyRef.current.slice(0, indexRef.current + 1);
        newHistory.push(value);
        if (newHistory.length > 50) {
          newHistory.shift(); // keep last 50
        } else {
          indexRef.current++;
        }
        historyRef.current = newHistory;
      }
    }, 400); // 400ms debounce
    return () => clearTimeout(timeoutId);
  }, [value]);

  const undo = useCallback((e?: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (indexRef.current > 0) {
      indexRef.current--;
      setValueState(historyRef.current[indexRef.current]);
    }
  }, []);

  const redo = useCallback((e?: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (indexRef.current < historyRef.current.length - 1) {
      indexRef.current++;
      setValueState(historyRef.current[indexRef.current]);
    }
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.ctrlKey || e.metaKey) {
      if (e.key === 'z') {
        if (e.shiftKey) {
          redo(e);
        } else {
          undo(e);
        }
      } else if (e.key === 'y') {
        redo(e);
      }
    }
  }, [undo, redo]);

  return { value, setValue, reset, undo, redo, handleKeyDown };
}
