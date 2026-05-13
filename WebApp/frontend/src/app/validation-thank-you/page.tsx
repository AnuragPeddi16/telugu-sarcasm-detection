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
              You have successfully validated <strong>{count}</strong> sample{parseInt(count) !== 1 ? 's' : ''}!
            </p>
            <p className={styles.messageTextTelugu}>
              మీరు విజయవంతంగా <strong>{count}</strong> నమూనా{parseInt(count) !== 1 ? 'లు' : ''} ధృవీకరించారు!
            </p>
          </div>

          <div className={styles.appreciation}>
            <p className={styles.appreciationText}>
              Your validation helps improve the accuracy of our Telugu sarcasm detection. 
              Each review you provide makes our dataset more reliable and comprehensive.
            </p>
            <p className={styles.appreciationTextTelugu}>
              మీ ధృవీకరణ మా తెలుగు వ్యంగ్య గుర్తింపు ఖచ్చితత్వాన్ని మెరుగుపరచడంలో సహాయపడుతుంది. 
              మీరు అందించే ప్రతి సమీక్ష మా డేటాసెట్‌ను మరింత విశ్వసనీయంగా మరియు సమగ్రంగా చేస్తుంది.
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

export default function ValidationThankYouPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ThankYouContent />
    </Suspense>
  )
}
