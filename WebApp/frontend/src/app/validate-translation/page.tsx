'use client'

import { useState, useRef, useEffect } from 'react'
import { useUndoHistory } from '@/utils/undoHistory'
//import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import DemographicForm from '@/components/DemographicForm'
import TeluguKeyboard from '@/components/TeluguKeyboard'
import Tutorial from '@/components/Tutorial'
import HistoryWidget, { HistoryItem } from '@/components/HistoryWidget'
import { api, type DemographicInfo, type TranslationSample } from '@/utils/api'
import { handleTransliterationInput, useGoogleInputTool, flushTransliteration } from '@/utils/transliteration'
import styles from './page.module.css'

type ResponseType = 'text' | 'audio'

export default function ValidateTranslationPage() {
  const router = useRouter()
  const [showTutorial, setShowTutorial] = useState(true)
  const [showDemographics, setShowDemographics] = useState(true)
  const [demographics, setDemographics] = useState<DemographicInfo | null>(null)
  const [sample, setSample] = useState<TranslationSample | null>(null)
  
  // History and Editing State
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [isReviewing, setIsReviewing] = useState(false)
  const [wasReviewingBeforeEdit, setWasReviewingBeforeEdit] = useState(false) // Track origin of edit

  const [isSarcastic, setIsSarcastic] = useState<boolean | null>(null)
  const { value: sarcasticExample, setValue: setSarcasticExample, reset: resetSarcasticExample, handleKeyDown: handleKeyDownSarcasticExample } = useUndoHistory('')

  // Enable Google Input Tool ONLY when the textarea is actually mounted
  const isFormVisible = !showTutorial && !showDemographics && !isReviewing
  useGoogleInputTool('validation-example-input', isFormVisible)
  const [responseType, setResponseType] = useState<ResponseType>('text')
  const [isRecording, setIsRecording] = useState(false)
  const [showKeyboard, setShowKeyboard] = useState(false)
  const [loading, setLoading] = useState(false)
  const [completedCount, setCompletedCount] = useState(0)
  const [seenIds, setSeenIds] = useState<string[]>([])
  const [sessionGoldStats, setSessionGoldStats] = useState({ total: 0, correct: 0 })
  const MIN_SAMPLES = parseInt(process.env.NEXT_PUBLIC_MIN_TRANSLATION_SAMPLES || '5', 10)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  useEffect(() => {
    if (!showTutorial && !showDemographics && !sample && !isReviewing) {
      loadNewSample()
    }
  }, [showTutorial, showDemographics, sample, isReviewing])

  const loadNewSample = async () => {
    setLoading(true)
    try {
      // Logic: First sample gold, then every 5th sample gold
      // completedCount starts at 0. So 0 => Gold. 1..4 => Normal. 5 => Gold.
      const isGold = completedCount === 0 || (completedCount > 0 && completedCount % 5 === 0)
      
      const newSample = await api.getTranslationSample(seenIds, isGold)
      
      // If we got a gold sample but requested normal (or vice versa due to fallback), 
      // rely on isGold flag from backend if present, or our logic.
      // Backend sets isGold=True if it served a gold sample.
      
      setSample(newSample)
      setSeenIds(prev => [...prev, newSample.id])
      
      setIsSarcastic(null)
      resetSarcasticExample('')
      setResponseType('text')
      setEditingIndex(null)
    } catch (error) {
      console.error('Error loading sample:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleReport = async () => {
    if (!sample) return
    if (!confirm('Are you sure you want to report this translation as invalid/bad?')) return
    
    try {
      await api.reportTranslation(sample.id, 'user_reported')
      alert('Reported successfully. Loading next sample...')
      loadNewSample()
    } catch (error) {
      alert('Error reporting.')
    }
  }

  const handleTutorialStart = () => {
    setShowTutorial(false)
  }

  const handleDemographicSubmit = (data: DemographicInfo) => {
    setDemographics(data)
    setShowDemographics(false)
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const reader = new FileReader()
        reader.readAsDataURL(audioBlob)
        reader.onloadend = () => {
          const base64Audio = reader.result as string
          setSarcasticExample(base64Audio)
        }
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error starting recording:', error)
      alert('Could not access microphone. Please ensure microphone permissions are granted.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const handleTextInput = (value: string) => {
    handleTransliterationInput(value, setSarcasticExample)
  }

  const handleKeyboardInput = (char: string) => {
    if (char === 'BACKSPACE') {
      setSarcasticExample(prev => prev.slice(0, -1))
    } else {
      setSarcasticExample(prev => prev + char)
    }
  }

  const handleSubmit = async () => {
    if (!sample || !demographics || isSarcastic === null) return

    let finalSarcasticExample = sarcasticExample;
    if (!isSarcastic && responseType === 'text') {
      finalSarcasticExample = await flushTransliteration(sarcasticExample);
      setSarcasticExample(finalSarcasticExample);
    }

    // If marked as non-sarcastic, require an example
    if (!isSarcastic && !finalSarcasticExample.trim()) {
      alert('Please provide a sarcastic example / దయచేసి వ్యంగ్య ఉదాహరణ ఇవ్వండి')
      return
    }

    // Gold Check Logic
    let currentStats = sessionGoldStats;
    if (sample.isGold && sample.correctIsSarcastic !== undefined) {
      const isCorrectlyIdentified = (isSarcastic === sample.correctIsSarcastic)

      currentStats = {
        total: sessionGoldStats.total + 1,
        correct: sessionGoldStats.correct + (isCorrectlyIdentified ? 1 : 0)
      }
      setSessionGoldStats(currentStats)
    }

    setLoading(true)
    try {
      await api.submitValidationResponse({
        sampleId: sample.id,
        originalUtterance: sample.utterance,
        isSarcastic,
        sarcasticExample: !isSarcastic ? finalSarcasticExample : undefined,
        demographics,
        responseType: !isSarcastic ? responseType : undefined,
        audioData: !isSarcastic && responseType === 'audio' ? finalSarcasticExample : undefined,
        sessionGoldStats: currentStats
      })

        // Add to history
      const newItem: HistoryItem = {
        id: sample.id,
        originalContext: sample.utterance,
        isSarcastic,
        sarcasticExample: !isSarcastic ? finalSarcasticExample : undefined,
        type: 'validation'
      }

      setHistory(prev => {
        if (editingIndex !== null) {
          const newHistory = [...prev]
          newHistory[editingIndex] = newItem
          return newHistory
        }
        return [...prev, newItem]
      })

      // Only increment completed count for new submissions
      if (editingIndex === null) {
        setCompletedCount(prev => prev + 1)
      }

      // Handle navigation based on edit/review state
      if (editingIndex !== null) {
        setEditingIndex(null)
        if (wasReviewingBeforeEdit) {
           setIsReviewing(true)
           setSample(null)
        } else {
           // If editing from widget during task, create new task (or clear edit state)
           loadNewSample() 
        }
      } else {
        // Normal submission flow
        loadNewSample()
      }

    } catch (error) {
      console.error('Error submitting response:', error)
      alert('Error submitting response. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleSkip = () => {
    loadNewSample()
  }

  // Old handleQuit removed from here


  const handleEditHistoryTrigger = (item: HistoryItem, index: number) => {
    setWasReviewingBeforeEdit(isReviewing)
    handleEditHistory(item, index)
  }

  const handleEditHistory = (item: HistoryItem, index: number) => {
    setEditingIndex(index)
    
    // For validation, we need to reconstruct the sample object partially
    setSample({
      id: item.id,
      utterance: item.originalContext,
      displayCount: 0
    })
    
    setIsSarcastic(item.isSarcastic ?? null)
    resetSarcasticExample(item.sarcasticExample || '')
    
    setIsReviewing(false)
    setShowHistory(false)
  }

  const handleSaveAndExit = () => {
    // If we have history, show review page
    if (history.length > 0) {
      setIsReviewing(true)
      setSample(null)
    } else {
      // If nothing done, just go home
      router.push('/')
    }
  }

  const handleQuit = () => {
      if (confirm('Are you sure you want to quit? Your progress will be lost if you leave without saving.')) {
        router.push('/')
      }
  }

  const handleFinalSubmit = () => {
    if (completedCount < MIN_SAMPLES) {
      alert(`Please validate at least ${MIN_SAMPLES} translations before finishing. You have validated ${completedCount}.`)
      return
    }
    router.push(`/validation-thank-you?count=${completedCount}&correct=${sessionGoldStats.correct}`)
  }

  if (showTutorial) {
    return <Tutorial onStart={handleTutorialStart} type="validation" />
  }

  if (showDemographics) {
    return <DemographicForm onSubmit={handleDemographicSubmit} onBack={() => router.push('/')} />
  }

  // Review Mode View
  if (isReviewing) {
    return (
      <main className={styles.main}>
        <div className={styles.container}>
          <div className={styles.header}>
            <h2 className={styles.reviewTitle}>Review Your Validations / మీ ధృవీకరణలను సమీక్షించండి</h2>
          </div>
          
          <div className={styles.reviewList}>
            {history.map((item, index) => (
              <div key={item.id} className={styles.reviewCard}>
                <div className={styles.reviewCardHeader}>
                  <span className={styles.reviewIndex}>#{index + 1}</span>
                  <button 
                    className={styles.editButton}
                    onClick={() => {
                        setWasReviewingBeforeEdit(true)
                        handleEditHistory(item, index)
                    }}
                  >
                    Edit / సవరించు
                  </button>
                </div>
                <div className={styles.reviewContent}>
                    <p className={styles.reviewContext}>
                          <strong>Utterance / వ్యాఖ్య:</strong> {item.originalContext.replace(/\\n/g, '\n').split('\n').map((line, i) => <div key={i}>{line}</div>)}
                    </p>
                    <div className={styles.reviewResponseGroup}>
                       <div className={styles.reviewResponse}>
                          <span className={styles.responseLabel}>Verdict:</span>
                          <span className={styles.responseText}>{item.isSarcastic ? 'Sarcastic' : 'Non-Sarcastic'}</span>
                       </div>
                       {!item.isSarcastic && item.sarcasticExample && (
                        <div className={styles.reviewResponse}>
                            <span className={styles.responseLabel}>Example:</span>
                            <span className={styles.responseText}>{item.sarcasticExample}</span>
                        </div>
                       )}
                    </div>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.actionButtons}>
            <button 
              className={styles.secondaryButton} 
              onClick={() => {
                setIsReviewing(false)
                setWasReviewingBeforeEdit(false)
                loadNewSample()
              }}
            >
              Add More / మరిన్ని జోడించండి
            </button>
            <button className={styles.primaryButton} onClick={handleFinalSubmit}>
              Confirm & Submit / నిర్ధారించండి & సమర్పించండి
            </button>
          </div>
        </div>
      </main>
    )
  }

  const canSubmit = isSarcastic !== null && (isSarcastic || sarcasticExample.trim().length > 0)

  return (
    <main className={styles.main}>
      <HistoryWidget 
        history={history} 
        isOpen={showHistory} 
        onClose={() => setShowHistory(false)} 
        onEdit={(item, index) => {
          setWasReviewingBeforeEdit((prev) => isReviewing)
          handleEditHistory(item, index)
        }}
        theme="orange"
      />

      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.progress}>
            Completed: {completedCount} / పూర్తైంది: {completedCount}
          </div>
          <div className={styles.headerButtons}>
            <button className={styles.historyButton} onClick={() => setShowHistory(true)}>
              History / చరిత్ర ({history.length})
            </button>
            <button className={styles.saveExitButton} onClick={handleSaveAndExit}>
                Save & Exit / సేవ్ & ఎగ్జిట్
            </button>
            <button className={styles.quitButton} onClick={handleQuit}>
               Quit / నిష్క్రమించు
            </button>
          </div>
        </div>

        {loading ? (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Loading... / లోడ్ అవుతోంది...</p>
          </div>
        ) : sample ? (
          <div className={styles.content}>
              <div className={styles.responseCard}>
                <h3 className={styles.responseLabel}>
                  Utterance
                  <span className={styles.responseLabelTelugu}>ప్రతిస్పందన</span>
                </h3>
                <div className={styles.highlightedResponse}>
                  {sample.utterance.replace(/\\n/g, '\n').split('\n').map((line, index) => (
                    <div key={index} className={styles.conversationLine}>
                      {line}
                    </div>
                  ))}
                </div>
                <button
                  onClick={handleReport}
                style={{
                  marginTop: '10px',
                  backgroundColor: '#dc3545',
                  color: 'white',
                  border: 'none',
                  padding: '5px 10px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.8rem'
                }}
              >
                Report Bad Translation / నివేదించండి
              </button>
            </div>

            <div className={styles.validationCard}>
              <h3 className={styles.questionTitle}>
                Is this response sarcastic?
                <span className={styles.questionTitleTelugu}>ఈ ప్రతిస్పందన వ్యంగ్యమా?</span>
              </h3>

              <div className={styles.binaryOptions}>
                <button
                  className={`${styles.optionButton} ${styles.sarcasticButton} ${isSarcastic === true ? styles.selected : ''}`}
                  onClick={() => {
                    setIsSarcastic(true)
                    resetSarcasticExample('')
                  }}
                >
                  ✓ Sarcastic
                  <span className={styles.optionButtonTelugu}>వ్యంగ్యం</span>
                </button>
                <button
                  className={`${styles.optionButton} ${styles.nonSarcasticButton} ${isSarcastic === false ? styles.selected : ''}`}
                  onClick={() => setIsSarcastic(false)}
                >
                  ✗ Non-Sarcastic
                  <span className={styles.optionButtonTelugu}>వ్యంగ్యం కాదు</span>
                </button>
              </div>

              {isSarcastic === false && (
                <div className={styles.exampleSection}>
                  <h4 className={styles.exampleTitle}>
                    Please provide a sarcastic example
                    <span className={styles.exampleTitleTelugu}>దయచేసి వ్యంగ్య ఉదాహరణ ఇవ్వండి</span>
                  </h4>

                  <div className={styles.inputTypeSelector}>
                    <button
                      className={`${styles.inputTypeButton} ${responseType === 'text' ? styles.activeInputType : ''}`}
                      onClick={() => {
                        setResponseType('text')
                        setIsRecording(false)
                      }}
                    >
                      📝 Text / వచనం
                    </button>
                    <button
                      className={`${styles.inputTypeButton} ${responseType === 'audio' ? styles.activeInputType : ''}`}
                      onClick={() => setResponseType('audio')}
                    >
                      🎤 Voice / వాయిస్
                    </button>
                  </div>

                  {responseType === 'text' ? (
                    <div className={styles.textInputContainer}>
                      <textarea
                        id="validation-example-input"
                        className={styles.textarea}
                        value={sarcasticExample}
                        onChange={(e) => handleTextInput(e.target.value)}
                        placeholder="Type a sarcastic version here / వ్యంగ్య సంస్కరణ ఇక్కడ టైప్ చేయండి"
                        rows={3}
                      />
                      <div className={styles.inputTools}>
                        <button
                          className={styles.keyboardButton}
                          onClick={() => setShowKeyboard(true)}
                        >
                          ⌨️ Telugu Keyboard / తెలుగు కీబోర్డ్
                        </button>
                        <p className={styles.hint}>
                          Tip: Type in English and press space to convert to Telugu
                          <br />
                          <span className={styles.hintTelugu}>
                            చిట్కా: ఆంగ్లంలో టైప్ చేసి స్పేస్ నొక్కండి తెలుగులోకి మారుతుంది
                          </span>
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.voiceInputContainer}>
                      {isRecording ? (
                        <button
                          className={`${styles.recordButton} ${styles.recording}`}
                          onClick={stopRecording}
                        >
                          <div className={styles.recordingIndicator}></div>
                          Stop Recording / రికార్డింగ్ ఆపండి
                        </button>
                      ) : (
                        <button
                          className={styles.recordButton}
                          onClick={startRecording}
                        >
                          🎤 {sarcasticExample ? 'Record/Re-record (రికార్డింగ్ మళ్ళీ చేయండి)' : 'Start Recording / రికార్డింగ్ ప్రారంభించండి'}
                        </button>
                      )}
                      {sarcasticExample && !isRecording && (
                        <div className={styles.recordedIndicator}>
                          ✓ Recording saved / రికార్డింగ్ సేవ్ చేయబడింది
                        <br />
                        <audio src={sarcasticExample} controls style={{ marginTop: '10px', width: '100%' }} />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className={styles.actions}>
              <button
                className={styles.skipButton}
                onClick={handleSkip}
                disabled={loading}
              >
                Skip / దాటవేయి
              </button>
              <button
                className={styles.submitButton}
                onClick={handleSubmit}
                disabled={!canSubmit || loading}
              >
                {editingIndex !== null ? 'Save Changes / మార్పులను సేవ్ చేయండి' : 'Next / తదుపరి'}
              </button>
            </div>
          </div>
        ) : null}

        {showKeyboard && (
          <TeluguKeyboard
            onInput={handleKeyboardInput}
            onClose={() => setShowKeyboard(false)}
            value={sarcasticExample}
            onChange={handleTextInput}
          />
        )}
      </div>
    </main>
  )
}
