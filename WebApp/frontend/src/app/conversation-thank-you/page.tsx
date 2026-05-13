'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { Suspense } from 'react'
import styles from './page.module.css'

function ThankYouContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const count = searchParams.get('count') || '0'

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <div className={styles.card}>
          <div className={styles.icon}>✓</div>
          <h1 className={styles.title}>Thank You!</h1>
          <p className={styles.titleTelugu}>ధన్యవాదాలు!</p>
          
          <div className={styles.message}>
            <p className={styles.messageText}>
              You have successfully completed <strong>{count}</strong> conversation{parseInt(count) !== 1 ? 's' : ''}!
            </p>
            <p className={styles.messageTextTelugu}>
              మీరు విజయవంతంగా <strong>{count}</strong> సంభాషణ{parseInt(count) !== 1 ? 'లు' : ''} పూర్తి చేశారు!
            </p>
          </div>

          <div className={styles.appreciation}>
            <p className={styles.appreciationText}>
              Your contribution helps us build better Telugu language technology. 
              Every response you provide makes our AI more culturally aware and accurate.
            </p>
            <p className={styles.appreciationTextTelugu}>
              మీ సహకారం మెరుగైన తెలుగు భాషా సాంకేతికతను రూపొందించడంలో మాకు సహాయపడుతుంది. 
              మీరు అందించే ప్రతి ప్రతిస్పందన మా AIని మరింత సాంస్కృతికంగా అవగాహన మరియు ఖచ్చితత్వంతో చేస్తుంది.
            </p>
          </div>

          <button className={styles.homeButton} onClick={() => router.push('/')}>
            Back to Home / హోమ్‌కు తిరిగి
          </button>
        </div>
      </div>
    </main>
  )
}

export default function ConversationThankYouPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ThankYouContent />
    </Suspense>
  )
}
