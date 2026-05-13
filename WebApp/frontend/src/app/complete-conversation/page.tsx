'use client'

import { useState, useRef, useEffect } from 'react'
import { useUndoHistory } from '@/utils/undoHistory'
//import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import DemographicForm from '@/components/DemographicForm'
import TeluguKeyboard from '@/components/TeluguKeyboard'
import Tutorial from '@/components/Tutorial'
import HistoryWidget, { HistoryItem } from '@/components/HistoryWidget'
import { api, type DemographicInfo, type ConversationSample } from '@/utils/api'
import { handleTransliterationInput, useGoogleInputTool, flushTransliteration } from '@/utils/transliteration'
import styles from './page.module.css'

type ResponseType = 'text' | 'audio'
type Stage = 'sarcastic' | 'non-sarcastic'

export default function CompleteConversationPage() {
  const router = useRouter()
  const [showTutorial, setShowTutorial] = useState(true)
  const [showDemographics, setShowDemographics] = useState(true)
  const [demographics, setDemographics] = useState<DemographicInfo | null>(null)
  const [conversation, setConversation] = useState<ConversationSample | null>(null)
  
  // History and Editing State
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [isReviewing, setIsReviewing] = useState(false)
  const [wasReviewingBeforeEdit, setWasReviewingBeforeEdit] = useState(false)

  const [currentStage, setCurrentStage] = useState<Stage>('sarcastic')
  
  // Enable Google Input Tool ONLY when we are actually showing the conversation form
  const isFormVisible = !showTutorial && !showDemographics && !isReviewing
  useGoogleInputTool('conversation-response-input', isFormVisible)
  const { value: sarcasticResponse, setValue: setSarcasticResponse, reset: resetSarcastic, handleKeyDown: handleKeyDownSarcastic } = useUndoHistory('')
  const { value: nonSarcasticResponse, setValue: setNonSarcasticResponse, reset: resetNonSarcastic, handleKeyDown: handleKeyDownNonSarcastic } = useUndoHistory('')
  const [responseType, setResponseType] = useState<ResponseType>('text')
  const [isRecording, setIsRecording] = useState(false)
  const [showKeyboard, setShowKeyboard] = useState(false)
  const [loading, setLoading] = useState(false)
  const [completedCount, setCompletedCount] = useState(0)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    // Only load new conversation if:
    // 1. Not showing tutorial or demographics
    // 2. No active conversation
    // 3. Not in Review Mode
    // 4. Not currently editing an item (editingIndex is null)
    if (!showTutorial && !showDemographics && !conversation && !isReviewing && editingIndex === null) {
      loadNewConversation()
    }
  }, [showTutorial, showDemographics, conversation, isReviewing, editingIndex])

  const loadNewConversation = async () => {
    setLoading(true)
    try {
      const sample = await api.getConversationSample()
      setConversation(sample)
      resetSarcastic('')
      resetNonSarcastic('')
      setCurrentStage('sarcastic')
      setResponseType('text')
      setEditingIndex(null)
    } catch (error) {
      console.error('Error loading conversation:', error)
    } finally {
      setLoading(false)
    }
  }

  const MIN_SAMPLES = parseInt(process.env.NEXT_PUBLIC_MIN_CONVERSATION_SAMPLES || '5', 10)

  const handleFinalSubmit = () => {
      if (history.length < MIN_SAMPLES) {
          alert(`Please complete at least ${MIN_SAMPLES} conversations before submitting. You have completed ${history.length}.`)
          return
      }
      router.push(`/conversation-thank-you?count=${completedCount}`)
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
        // Convert to base64 for storage/transmission
        const reader = new FileReader()
        reader.readAsDataURL(audioBlob)
        reader.onloadend = () => {
          const base64Audio = reader.result as string
          if (currentStage === 'sarcastic') {
            setSarcasticResponse(base64Audio)
          } else {
            setNonSarcasticResponse(base64Audio)
          }
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
    if (currentStage === 'sarcastic') {
      handleTransliterationInput(value, setSarcasticResponse)
    } else {
      handleTransliterationInput(value, setNonSarcasticResponse)
    }
  }

  const handleKeyboardInput = (char: string) => {
    if (char === 'BACKSPACE') {
      if (currentStage === 'sarcastic') {
        setSarcasticResponse(prev => prev.slice(0, -1))
      } else {
        setNonSarcasticResponse(prev => prev.slice(0, -1))
      }
    } else {
      if (currentStage === 'sarcastic') {
        setSarcasticResponse(prev => prev + char)
      } else {
        setNonSarcasticResponse(prev => prev + char)
      }
    }
  }

  const handleNext = async () => {
    let finalSarcastic = sarcasticResponse;
    let finalNonSarcastic = nonSarcasticResponse;

    if (currentStage === 'sarcastic' && responseType === 'text') {
      finalSarcastic = await flushTransliteration(sarcasticResponse);
      setSarcasticResponse(finalSarcastic);
    } else if (currentStage === 'non-sarcastic' && responseType === 'text') {
      finalNonSarcastic = await flushTransliteration(nonSarcasticResponse);
      setNonSarcasticResponse(finalNonSarcastic);
    }

    if (currentStage === 'sarcastic') {
      setCurrentStage('non-sarcastic')
      setResponseType('text')
    } else {
      await handleSubmit(finalSarcastic, finalNonSarcastic)
    }
  }

  const handleSubmit = async (finalSarcastic: string = sarcasticResponse, finalNonSarcastic: string = nonSarcasticResponse) => {
    if (!conversation || !demographics) return

    setLoading(true)
    try {
      const responseData = {
        sampleId: conversation.id,
        originalConversation: conversation.conversation,
        sarcasticResponse: finalSarcastic,
        nonSarcasticResponse: finalNonSarcastic,
        demographics,
        responseType,
        audioData: responseType === 'audio' ?
          (currentStage === 'sarcastic' ? finalSarcastic : finalNonSarcastic) :
          undefined
      }

      await api.submitConversationResponse(responseData)

      // Update history
      const newItem: HistoryItem = {
        id: conversation.id,
        originalContext: conversation.conversation,
        sarcasticResponse: finalSarcastic,
        nonSarcasticResponse: finalNonSarcastic,
        type: 'conversation'
      }

      setHistory(prev => {
        if (editingIndex !== null) {
          const newHistory = [...prev]
          newHistory[editingIndex] = newItem
          return newHistory
        }
        return [...prev, newItem]
      })

      // ONLY increment if adding new item
      if (editingIndex === null) {
        setCompletedCount(prev => prev + 1)
      }
      
      // Navigation Logic after Save
      if (editingIndex !== null) {
        // Editing Mode
        if (wasReviewingBeforeEdit) {
          // If came from Review Page, go back to Review Page
          setIsReviewing(true)
          setConversation(null) 
        } else {
          // If came from History Widget (while doing task), go back to NEW task
          loadNewConversation()
        }
        setEditingIndex(null)
        setWasReviewingBeforeEdit(false) // Reset
      } else {
        // Normal Flow (Added new item) -> Go to next new item
        loadNewConversation()
      }

    } catch (error) {
      console.error('Error submitting response:', error)
      alert('Error submitting response. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleSkip = () => {
    loadNewConversation()
  }

  const handleSaveAndExit = () => {
    // If there is history, show review page
    if (history.length > 0) {
      setIsReviewing(true)
      setConversation(null)
    } else {
      router.push('/')
    }
  }

  const handleQuit = () => {
    const confirmed = window.confirm(
      "Are you sure you want to quit? Passible unsaved changes will be lost.\n\nమీరు నిజంగా నిష్క్రమించాలనుకుంటున్నారా?"
    )
    if (confirmed) {
      router.push('/')
    }
  }

  const handleEditHistory = (item: HistoryItem, index: number) => {
    // Track where we came from
    setWasReviewingBeforeEdit(isReviewing)
    
    // Load item back into form
    setEditingIndex(index)
    setConversation({
      id: item.id,
      conversation: item.originalContext,
      displayCount: 0 // Not relevant for edit
    })
    resetSarcastic(item.sarcasticResponse || '')
    resetNonSarcastic(item.nonSarcasticResponse || '')
    setCurrentStage('sarcastic')
    setIsReviewing(false) // Exit review mode to show form
    setShowHistory(false) // Close widget if open
  }

  if (showTutorial) {
    return <Tutorial onStart={handleTutorialStart} type="conversation" />
  }

  if (showDemographics) {
    return <DemographicForm onSubmit={handleDemographicSubmit} onBack={() => router.push('/')} />
  }

  const handleEditHistoryTrigger = (item: HistoryItem, index: number) => {
    setWasReviewingBeforeEdit((prev) => isReviewing) // Use current state
    handleEditHistory(item, index)
  }

  // Review Mode View
  if (isReviewing) {
    return (
      <main className={styles.main}>
        <div className={styles.container}>
          <div className={styles.header}>
            <h2 className={styles.reviewTitle}>Review Your Responses / మీ సమాధానాలను సమీక్షించండి</h2>
          </div>
          
          <div className={styles.reviewList}>
            {history.map((item, index) => (
              <div key={item.id} className={styles.reviewCard}>
                <div className={styles.reviewCardHeader}>
                  <span className={styles.reviewIndex}>#{index + 1}</span>
                  <button 
                    className={styles.editButton}
                    onClick={() => handleEditHistoryTrigger(item, index)}
                  >
                    Edit / సవరించు
                  </button>
                </div>
                <div className={styles.reviewContent}>
                    <p className={styles.reviewContext}>
                          <strong>Context:</strong> {item.originalContext.replace(/\\n/g, '\n').split('\n').map((line, i) => <div key={i}>{line}</div>)}
                    </p>
                    <div className={styles.reviewResponseGroup}>
                        <div className={styles.reviewResponse}>
                        <span className={styles.responseLabel}>Sarcastic:</span> 
                        {item.sarcasticResponse ? (
                           <span className={styles.responseText}>{item.sarcasticResponse}</span>
                        ) : (
                           <div className={styles.audioPlayer}>
                             (Audio Review Not Implemented)
                           </div>
                        )}
                        </div>
                        <div className={styles.reviewResponse}>
                        <span className={styles.responseLabel}>Non-Sarcastic:</span> 
                        {item.nonSarcasticResponse ? (
                           <span className={styles.responseText}>{item.nonSarcasticResponse}</span>
                        ) : (
                            <div className={styles.audioPlayer}>
                             (Audio Review Not Implemented)
                           </div>
                        )}
                        </div>
                    </div>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.actionButtons}>
             {/* Allow going back to add more if needed */}
            <button 
              className={styles.secondaryButton} 
              onClick={() => {
                setIsReviewing(false)
                setWasReviewingBeforeEdit(false)
                loadNewConversation()
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

  // Ensure conversation is loaded before rendering form
  if (!conversation) return <div className={styles.loading}>Loading...</div>

  const currentResponse = currentStage === 'sarcastic' ? sarcasticResponse : nonSarcasticResponse
  const isResponseValid = responseType === 'audio' ? currentResponse.length > 0 : currentResponse.trim().length > 0

  return (
    <main className={styles.main}>
      <HistoryWidget 
        history={history} 
        isOpen={showHistory} 
        onClose={() => setShowHistory(false)} 
        onEdit={handleEditHistoryTrigger}
        theme="blue"
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
        ) : conversation ? (
          <div className={styles.content}>
            <div className={styles.stageIndicator}>
              <div className={`${styles.stageBox} ${currentStage === 'sarcastic' ? styles.activeStage : styles.completedStage}`}>
                1. Sarcastic Response / వ్యంగ్య ప్రతిస్పందన
              </div>
              <div className={`${styles.stageBox} ${currentStage === 'non-sarcastic' ? styles.activeStage : ''}`}>
                2. Non-Sarcastic Response / వ్యంగ్యం కాని ప్రతిస్పందన
              </div>
            </div>

            <div className={styles.conversationCard}>
              <h2 className={styles.cardTitle}>
                Conversation Starter
                <span className={styles.cardTitleTelugu}>సంభాషణ ప్రారంభం</span>
              </h2>
              <div className={styles.conversationText}>
                {conversation.conversation.replace(/\\n/g, '\n').split('\n').map((line, index) => (
                  <div key={index} className={styles.conversationLine}>
                    {line}
                  </div>
                ))}
              </div>
            </div>

            <div className={styles.responseCard}>
              <h3 className={styles.responseTitle}>
                {currentStage === 'sarcastic' 
                  ? 'Provide a Sarcastic Response'
                  : 'Provide a Non-Sarcastic Response'}
              </h3>
              <p className={styles.responseTitleTelugu}>
                {currentStage === 'sarcastic'
                  ? 'వ్యంగ్య ప్రతిస్పందన ఇవ్వండి'
                  : 'వ్యంగ్యం కాని ప్రతిస్పందన ఇవ్వండి'}
              </p>

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
                    id="conversation-response-input"
                    ref={textareaRef}
                    className={styles.textarea}
                    value={currentResponse}
                    onChange={(e) => handleTextInput(e.target.value)}
                    placeholder="Type your response here (you can type in English and it will convert to Telugu) / మీ ప్రతిస్పందనను ఇక్కడ టైప్ చేయండి"
                    rows={4}
                      onKeyDown={currentStage === 'sarcastic' ? handleKeyDownSarcastic : handleKeyDownNonSarcastic}
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
                      🎤 {currentResponse ? 'Record/Re-record (రికార్డింగ్ మళ్ళీ చేయండి)' : 'Start Recording / రికార్డింగ్ ప్రారంభించండి'}
                    </button>
                  )}
                  {currentResponse && !isRecording && (
                    <div className={styles.recordedIndicator}>
                      ✓ Recording saved / రికార్డింగ్ సేవ్ చేయబడింది
                        <br />
                        <audio src={currentResponse} controls style={{ marginTop: '10px', width: '100%' }} />
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
                className={styles.nextButton}
                onClick={handleNext}
                disabled={!isResponseValid || loading}
              >
                {currentStage === 'sarcastic' 
                  ? 'Next / తదుపరి' 
                  : (editingIndex !== null ? 'Save Changes / మార్పులను సేవ్ చేయండి' : 'Next / తదుపరి')}
              </button>
            </div>
          </div>
        ) : null}

        {showKeyboard && (
          <TeluguKeyboard
            onInput={handleKeyboardInput}
            onClose={() => setShowKeyboard(false)}
            value={currentStage === 'sarcastic' ? sarcasticResponse : nonSarcasticResponse}
            onChange={handleTextInput}
          />
        )}
      </div>
    </main>
  )
}
