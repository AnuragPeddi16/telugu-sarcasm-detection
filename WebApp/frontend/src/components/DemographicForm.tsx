'use client'

import { useState } from 'react'
import { districts, ageGroups, educationLevels, genders } from '@/utils/constants'
import styles from './DemographicForm.module.css'

interface DemographicFormProps {
  onSubmit: (data: {
    region: string
    gender: string
    education: string
    ageGroup: string
  }) => void
  onBack?: () => void
}

export default function DemographicForm({ onSubmit, onBack }: DemographicFormProps) {
  const [formData, setFormData] = useState({
    region: '',
    gender: '',
    education: '',
    ageGroup: ''
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (formData.region && formData.gender && formData.education && formData.ageGroup) {
      onSubmit(formData)
    }
  }

  const isFormValid = formData.region && formData.gender && formData.education && formData.ageGroup

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <div className={styles.card}>
          {onBack && (
            <button className={styles.backButton} onClick={onBack}>
              ← Back / వెనక్కి
            </button>
          )}
        <h2 className={styles.title}>Tell Us About Yourself</h2>
        <p className={styles.titleTelugu}>మీ గురించి మాకు చెప్పండి</p>
        <p className={styles.subtitle}>
          This information helps us understand diverse perspectives
        </p>
        <p className={styles.subtitleTelugu}>
          ఈ సమాచారం విభిన్న దృక్కోణాలను అర్థం చేసుకోవడంలో మాకు సహాయపడుతుంది
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label className={styles.label}>
              Region (District)
              <span className={styles.labelTelugu}>ప్రాంతం (జిల్లా)</span>
            </label>
            <select
              className={styles.select}
              value={formData.region}
              onChange={(e) => setFormData({ ...formData, region: e.target.value })}
              required
            >
              <option value="">Select your district / జిల్లా ఎంచుకోండి</option>
              {districts.map((district) => (
                <option key={district.value} value={district.value}>
                  {district.label}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>
              Age Group
              <span className={styles.labelTelugu}>వయస్సు సమూహం</span>
            </label>
            <div className={styles.radioGrid}>
              {ageGroups.map((age) => (
                <label key={age} className={styles.radioLabel}>
                  <input
                    type="radio"
                    name="ageGroup"
                    value={age}
                    checked={formData.ageGroup === age}
                    onChange={(e) => setFormData({ ...formData, ageGroup: e.target.value })}
                    className={styles.radio}
                  />
                  <span className={styles.radioText}>{age}</span>
                </label>
              ))}
            </div>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>
              Education
              <span className={styles.labelTelugu}>విద్య</span>
            </label>
            <select
              className={styles.select}
              value={formData.education}
              onChange={(e) => setFormData({ ...formData, education: e.target.value })}
              required
            >
              <option value="">Select education level / విద్యా స్థాయిని ఎంచుకోండి</option>
              {educationLevels.map((level) => (
                <option key={level.value} value={level.value}>
                  {level.label}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>
              Gender
              <span className={styles.labelTelugu}>లింగం</span>
            </label>
            <div className={styles.radioGrid}>
              {genders.map((gender) => (
                <label key={gender.value} className={styles.radioLabel}>
                  <input
                    type="radio"
                    name="gender"
                    value={gender.value}
                    checked={formData.gender === gender.value}
                    onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                    className={styles.radio}
                  />
                  <span className={styles.radioText}>{gender.label}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            type="submit"
            className={styles.submitButton}
            disabled={!isFormValid}
          >
            Continue / కొనసాగించు
          </button>
        </form>
      </div>
      </div>
    </main>
  )
}
