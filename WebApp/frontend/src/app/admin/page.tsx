'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import styles from './page.module.css'
import { api } from '@/utils/api'

interface Stats {
  totalResponses: number
  conversationResponses: number
  validationResponses: number
  sarcasticCount: number
  nonSarcasticCount: number
  demographics: {
    regions: { [key: string]: number }
    ageGroups: { [key: string]: number }
    education: { [key: string]: number }
    gender: { [key: string]: number }
  }
}

export default function AdminDashboard() {
  const router = useRouter()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(false)

  // Mock authentication
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const dbStats = await api.getAdminStats({ username, password })
      setStats(dbStats)
      setIsAuthenticated(true)
    } catch (err: any) {
      // Typically catching Axios error or generic Error
      if (err?.response?.status === 401 || err.message === 'Invalid credentials') {
        setError('Invalid credentials / చెల్లని ఆధారాలు')
      } else {
        setError('Error fetching stats / గణాంకాలను పొందడంలో లోపం')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setUsername('')
    setPassword('')
    setStats(null)
  }

  if (!isAuthenticated) {
    return (
      <main className={styles.main}>
        <div className={styles.loginContainer}>
          <div className={styles.loginCard}>
            <h1 className={styles.loginTitle}>Admin Dashboard</h1>
            <p className={styles.loginTitleTelugu}>అడ్మిన్ డాష్‌బోర్డ్</p>
            
            <form onSubmit={handleLogin} className={styles.loginForm}>
              <div className={styles.formGroup}>
                <label className={styles.label}>
                  Username / వినియోగదారు పేరు
                </label>
                <input
                  type="text"
                  className={styles.input}
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter username"
                  required
                />
              </div>

              <div className={styles.formGroup}>
                <label className={styles.label}>
                  Password / పాస్‌వర్డ్
                </label>
                <input
                  type="password"
                  className={styles.input}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  required
                />
              </div>

              {error && <div className={styles.error}>{error}</div>}

              <button type="submit" className={styles.loginButton} disabled={loading}>
                {loading ? 'Logging in... / లాగిన్ అవుతోంది...' : 'Login / లాగిన్'}
              </button>

              <button 
                type="button" 
                className={styles.backButton}
                onClick={() => router.push('/')}
              >
                Back to Home / హోమ్‌కు తిరిగి
              </button>
            </form>

            {/* <div className={styles.hint}>
              <p>Demo credentials:</p>
              <p>Username: admin</p>
              <p>Password: admin123</p>
            </div> */}
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className={styles.main}>
      <div className={styles.dashboardContainer}>
        <div className={styles.dashboardHeader}>
          <div>
            <h1 className={styles.dashboardTitle}>Admin Dashboard</h1>
            <p className={styles.dashboardTitleTelugu}>అడ్మిన్ డాష్‌బోర్డ్</p>
          </div>
          <button className={styles.logoutButton} onClick={handleLogout}>
            Logout / లాగ్అవుట్
          </button>
        </div>

        {loading ? (
          <div className={styles.loading}>
            <div className={styles.spinner}></div>
            <p>Loading statistics... / గణాంకాలు లోడ్ అవుతున్నాయి...</p>
          </div>
        ) : stats ? (
          <>
            <div className={styles.statsGrid}>
              <div className={`${styles.statCard} ${styles.primary}`}>
                <div className={styles.statNumber}>{stats.totalResponses}</div>
                <div className={styles.statLabel}>
                  Total Responses
                  <span>మొత్తం ప్రతిస్పందనలు</span>
                </div>
              </div>

              <div className={`${styles.statCard} ${styles.success}`}>
                <div className={styles.statNumber}>{stats.conversationResponses}</div>
                <div className={styles.statLabel}>
                  Conversations
                  <span>సంభాషణలు</span>
                </div>
              </div>

              <div className={`${styles.statCard} ${styles.info}`}>
                <div className={styles.statNumber}>{stats.validationResponses}</div>
                <div className={styles.statLabel}>
                  Validations
                  <span>ధృవీకరణలు</span>
                </div>
              </div>

              <div className={`${styles.statCard} ${styles.warning}`}>
                <div className={styles.statNumber}>{stats.sarcasticCount}</div>
                <div className={styles.statLabel}>
                  Sarcastic
                  <span>వ్యంగ్యం</span>
                </div>
              </div>

              <div className={`${styles.statCard} ${styles.danger}`}>
                <div className={styles.statNumber}>{stats.nonSarcasticCount}</div>
                <div className={styles.statLabel}>
                  Non-Sarcastic
                  <span>వ్యంగ్యం కాదు</span>
                </div>
              </div>

              <div className={`${styles.statCard} ${styles.secondary}`}>
                <div className={styles.statNumber}>
                  {stats.validationResponses > 0 ? ((stats.sarcasticCount / stats.validationResponses) * 100).toFixed(1) : 0}%
                </div>
                <div className={styles.statLabel}>
                  Sarcasm Rate (Validations)
                  <span>వ్యంగ్య రేటు</span>
                </div>
              </div>
            </div>

            <div className={styles.chartsGrid}>
              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>
                  Region Distribution
                  <span>ప్రాంత పంపిణీ</span>
                </h3>
                <div className={styles.barChart}>
                  {Object.entries(stats.demographics.regions).map(([region, count]) => (
                    <div key={region} className={styles.barItem}>
                      <div className={styles.barLabel}>{region}</div>
                      <div className={styles.barContainer}>
                        <div 
                          className={styles.barFill}
                          style={{ width: `${(count / stats.totalResponses) * 100}%` }}
                        >
                          <span className={styles.barValue}>{count}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>
                  Age Group Distribution
                  <span>వయస్సు సమూహ పంపిణీ</span>
                </h3>
                <div className={styles.barChart}>
                  {Object.entries(stats.demographics.ageGroups).map(([age, count]) => (
                    <div key={age} className={styles.barItem}>
                      <div className={styles.barLabel}>{age}</div>
                      <div className={styles.barContainer}>
                        <div 
                          className={`${styles.barFill} ${styles.success}`}
                          style={{ width: `${(count / stats.totalResponses) * 100}%` }}
                        >
                          <span className={styles.barValue}>{count}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>
                  Education Level
                  <span>విద్యా స్థాయి</span>
                </h3>
                <div className={styles.barChart}>
                  {Object.entries(stats.demographics.education).map(([edu, count]) => (
                    <div key={edu} className={styles.barItem}>
                      <div className={styles.barLabel}>{edu}</div>
                      <div className={styles.barContainer}>
                        <div 
                          className={`${styles.barFill} ${styles.warning}`}
                          style={{ width: `${(count / stats.totalResponses) * 100}%` }}
                        >
                          <span className={styles.barValue}>{count}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className={styles.chartCard}>
                <h3 className={styles.chartTitle}>
                  Gender Distribution
                  <span>లింగ పంపిణీ</span>
                </h3>
                <div className={styles.pieChart}>
                  {Object.entries(stats.demographics.gender).map(([gender, count], index) => (
                    <div key={gender} className={styles.pieItem}>
                      <div className={`${styles.pieColor} ${styles[`color${index + 1}`]}`}></div>
                      <div className={styles.pieLabel}>
                        {gender}: <strong>{count}</strong> ({((count / stats.totalResponses) * 100).toFixed(1)}%)
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </main>
  )
}
