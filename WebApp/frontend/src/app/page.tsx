'use client'

import { useRouter } from 'next/navigation'
import styles from './page.module.css'

export default function Home() {
  const router = useRouter()

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className={styles.title}>
            Telugu Sarcasm Detection Database
          </h1>
          <p className={styles.titleTelugu}>
            తెలుగు వ్యంగ్య గుర్తింపు డేటాబేస్
          </p>
        </div>

        <div className={styles.motivation}>
          <p className={styles.motivationText}>
            We are building a comprehensive database to help computers understand sarcasm in Telugu language. 
            Your contribution will help improve natural language processing for Telugu and make technology more culturally aware.
          </p>
          <p className={styles.motivationTextTelugu}>
            తెలుగు భాషలో వ్యంగ్యాన్ని కంప్యూటర్‌లు అర్థం చేసుకోవడంలో సహాయపడే విస్తృత డేటాబేస్‌ను మేము నిర్మిస్తున్నాము. 
            మీ సహకారం తెలుగు కోసం సహజ భాషా ప్రాసెసింగ్‌ను మెరుగుపరచడానికి మరియు సాంకేతికతను మరింత సాంస్కృతికంగా అవగాహన కలిగి ఉండేలా చేయడానికి సహాయపడుతుంది.
          </p>
        </div>

        <div className={styles.optionsContainer}>
          <h2 className={styles.chooseText}>
            Choose Your Contribution Method
            <span className={styles.chooseTextTelugu}>మీ సహకార పద్ధతిని ఎంచుకోండి</span>
          </h2>

          <div className={styles.options}>
            <button 
              className={`${styles.optionCard} ${styles.optionComplete}`}
              onClick={() => router.push('/complete-conversation')}
            >
              <div className={styles.optionIcon}>💬</div>
              <h3 className={styles.optionTitle}>Complete Conversations</h3>
              <p className={styles.optionTitleTelugu}>సంభాషణలను పూర్తి చేయండి</p>
              <p className={styles.optionDescription}>
                Read conversation starters and provide both sarcastic and non-sarcastic responses
              </p>
              <p className={styles.optionDescriptionTelugu}>
                సంభాషణ ప్రారంభాలను చదివి వ్యంగ్య మరియు వ్యంగ్యం కాని ప్రతిస్పందనలు ఇవ్వండి
              </p>
            </button>

            <button 
              className={`${styles.optionCard} ${styles.optionValidate}`}
              onClick={() => router.push('/validate-translation')}
            >
              <div className={styles.optionIcon}>✓</div>
              <h3 className={styles.optionTitle}>Validate Translations</h3>
              <p className={styles.optionTitleTelugu}>అనువాదాలను ధృవీకరించండి</p>
              <p className={styles.optionDescription}>
                Review translated conversations and mark whether they are sarcastic or not
              </p>
              <p className={styles.optionDescriptionTelugu}>
                అనువదించిన సంభాషణలను సమీక్షించి అవి వ్యంగ్యమా కాదా అని గుర్తించండి
              </p>
            </button>
          </div>
        </div>

        <footer className={styles.footer}>
          <p>Thank you for contributing to Telugu language technology!</p>
          <p className={styles.footerTelugu}>తెలుగు భాషా సాంకేతికతకు మీ సహకారానికి ధన్యవాదాలు!</p>
          
          <button 
            className={styles.adminLink}
            onClick={() => router.push('/admin')}
          >
            Admin Dashboard
          </button>
        </footer>
      </div>
    </main>
  )
}
