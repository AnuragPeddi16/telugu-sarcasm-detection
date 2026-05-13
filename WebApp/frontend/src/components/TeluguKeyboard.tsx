'use client'

import { useState, useRef } from 'react'
import styles from './TeluguKeyboard.module.css'

interface TeluguKeyboardProps {
  onInput: (char: string) => void
  onClose: () => void
  value?: string
  onChange?: (value: string) => void
}

const teluguKeys = {
  vowels: ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఋ', 'ౠ', 'ఌ', 'ౡ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'అం', 'అః'],
  consonants: [
    ['క', 'ఖ', 'గ', 'ఘ', 'ఙ'],
    ['చ', 'ఛ', 'జ', 'ఝ', 'ఞ'],
    ['ట', 'ఠ', 'డ', 'ఢ', 'ణ'],
    ['త', 'థ', 'ద', 'ధ', 'న'],
    ['ప', 'ఫ', 'బ', 'భ', 'మ'],
    ['య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ', 'ళ', 'క్ష', 'ఱ']
  ],
  matras: ['ా', 'ి', 'ీ', 'ు', 'ూ', 'ృ', 'ౄ', 'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', '్'],
  otthulu: [
    ['్క', '్ఖ', '్గ', '్ఘ', '్ఙ'],
    ['్చ', '్ఛ', '్జ', '్ఝ', '్ఞ'],
    ['్ట', '్ఠ', '్డ', '్ఢ', '్ణ'],
    ['్త', '్థ', '్ద', '్ధ', '్న'],
    ['్ప', '్ఫ', '్బ', '్భ', '్మ'],
    ['్య', '్ర', '్ల', '్వ', '్శ', '్ష', '్స', '్హ', '్ళ', '్క్ష', '్ఱ']
  ],
  numbers: ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯'],
  special: ['।', '॥', ' ']
}

export default function TeluguKeyboard({ onInput, onClose, value, onChange }: TeluguKeyboardProps) {
  const [activeTab, setActiveTab] = useState<'consonants' | 'vowels' | 'matras' | 'otthulu' | 'numbers'>('consonants')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleKeyPress = (char: string) => {
    if (value !== undefined && onChange !== undefined && textareaRef.current) {
      const start = textareaRef.current.selectionStart
      const end = textareaRef.current.selectionEnd
      
      let newValue = ''
      let newCursorPos = start
      
      if (char === 'BACKSPACE') {
        if (start === end && start > 0) {
          newValue = value.slice(0, start - 1) + value.slice(end)
          newCursorPos = start - 1
        } else if (start !== end) {
          newValue = value.slice(0, start) + value.slice(end)
          newCursorPos = start
        } else {
          newValue = value
        }
      } else {
        newValue = value.slice(0, start) + char + value.slice(end)
        newCursorPos = start + char.length
      }

      onChange(newValue)
      
      // Keep focus and restore cursor position
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.focus()
          textareaRef.current.setSelectionRange(newCursorPos, newCursorPos)
        }
      }, 0)
    } else {
      onInput(char)
    }
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.keyboard} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <h3 className={styles.title}>Telugu Keyboard / తెలుగు కీబోర్డ్</h3>
          <button className={styles.closeButton} onClick={onClose}>✕</button>
        </div>

        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${activeTab === 'consonants' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('consonants')}
          >
            Consonants / హల్లులు
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'vowels' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('vowels')}
          >
            Vowels / అచ్చులు
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'matras' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('matras')}
          >
            Matras / మాత్రలు
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'otthulu' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('otthulu')}
          >
            Otthulu / ఒత్తులు
          </button>
          <button
            className={`${styles.tab} ${activeTab === 'numbers' ? styles.activeTab : ''}`}
            onClick={() => setActiveTab('numbers')}
          >
            Numbers / సంఖ్యలు
          </button>
        </div>

        {value !== undefined && onChange !== undefined && (
          <div className={styles.previewContainer} style={{ padding: '10px 15px', borderBottom: '1px solid var(--border-color, #e5e7eb)' }}>
            <textarea
              ref={textareaRef}
              autoFocus
              value={value}
              onChange={(e) => onChange(e.target.value)}
              className={styles.previewInput}
              rows={2}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid var(--border-color, #d1d5db)',
                fontSize: '1rem',
                resize: 'none',
                fontFamily: 'inherit',
                outline: 'none'
              }}
              placeholder="Type here... / ఇక్కడ టైప్ చేయండి..."
            />
          </div>
        )}

        <div className={styles.keyboardContent} onMouseDown={(e) => e.preventDefault()}>
          {activeTab === 'consonants' && (
            <div className={styles.consonantGrid}>
              {teluguKeys.consonants.map((row, i) => (
                <div key={i} className={styles.keyRow}>
                  {row.map((char) => (
                    <button
                      key={char}
                      className={styles.key}
                      onClick={() => handleKeyPress(char)}
                    >
                      {char}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}

          {activeTab === 'vowels' && (
            <div className={styles.vowelGrid}>
              {teluguKeys.vowels.map((char) => (
                <button
                  key={char}
                  className={styles.key}
                  onClick={() => handleKeyPress(char)}
                >
                  {char}
                </button>
              ))}
            </div>
          )}

          {activeTab === 'matras' && (
            <div className={styles.matraGrid}>
              {teluguKeys.matras.map((char) => (
                <button
                  key={char}
                  className={styles.key}
                  onClick={() => handleKeyPress(char)}
                >
                  {char}
                </button>
              ))}
            </div>
          )}

          {activeTab === 'numbers' && (
            <div className={styles.numberGrid}>
              {teluguKeys.numbers.map((char) => (
                <button
                  key={char}
                  className={styles.key}
                  onClick={() => handleKeyPress(char)}
                >
                  {char}
                </button>
              ))}
            </div>
          )}

          {activeTab === 'otthulu' && (
            <div className={styles.consonantGrid}>
              {teluguKeys.otthulu.map((row, i) => (
                <div key={`otthu-row-${i}`} className={styles.keyRow}>
                  {row.map((char) => (
                    <button
                      key={char}
                      className={styles.key}
                      onClick={() => handleKeyPress(char)}
                    >
                      {char}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className={styles.actionBar} onMouseDown={(e) => e.preventDefault()}>
          <button
            className={styles.actionButton}
            onClick={() => handleKeyPress(' ')}
          >
            Space / ఖాళీ
          </button>
          <button
            className={styles.actionButton}
            onClick={() => handleKeyPress('BACKSPACE')}
          >
            ← Backspace
          </button>
        </div>
      </div>
    </div>
  )
}
