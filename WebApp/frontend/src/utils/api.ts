import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface DemographicInfo {
  region: string
  gender: string
  education: string
  ageGroup: string
}

export interface ConversationSample {
  id: string
  conversation: string
  displayCount: number
}

export interface TranslationSample {
  id: string
  utterance: string
  displayCount: number
  isGold?: boolean
  correctIsSarcastic?: boolean
}

export interface ConversationResponse {
  sampleId: string
  originalConversation: string
  sarcasticResponse: string
  nonSarcasticResponse: string
  demographics: DemographicInfo
  responseType: 'text' | 'audio'
  audioData?: string
}

export interface SessionGoldStats {
  total: number
  correct: number
}

export interface ValidationResponse {
  sampleId: string
  originalUtterance: string
  isSarcastic: boolean
  sarcasticExample?: string
  demographics: DemographicInfo
  responseType?: 'text' | 'audio'
  audioData?: string
  sessionGoldStats?: SessionGoldStats
}

const USE_BACKEND = process.env.NEXT_PUBLIC_USE_BACKEND === 'true'

// Real API Implementation
const realApi = {
  async getConversationSample(seenIds: string[] = []): Promise<ConversationSample> {
    try {
      const params = new URLSearchParams();
      seenIds.forEach(id => params.append('seenIds', id));
      const response = await apiClient.get<ConversationSample>('/conversations/random', { params })
      return response.data
    } catch (error) {
      console.error('Error fetching conversation sample:', error)
      throw error
    }
  },

  async getTranslationSample(seenIds: string[] = [], isGold: boolean = false): Promise<TranslationSample> {
    try {
      const params = new URLSearchParams();
      seenIds.forEach(id => params.append('seenIds', id));
      if (isGold) params.append('isGold', 'true');
      const response = await apiClient.get<TranslationSample>('/validations/random', { params })
      return response.data
    } catch (error) {
      console.error('Error fetching translation sample:', error)
      throw error
    }
  },

  async reportTranslation(id: string, reason: string): Promise<{ success: boolean }> {
    try {
      await apiClient.post('/validations/report', { id, reason })
      return { success: true }
    } catch (error) {
      console.error('Error reporting translation:', error)
      return { success: false }
    }
  },

  async submitConversationResponse(data: ConversationResponse): Promise<{ success: boolean }> {
    try {
      const response = await apiClient.post('/conversations/submit', data)
      return { success: true }
    } catch (error) {
      console.error('Error submitting conversation response:', error)
      return { success: false }
    }
  },

  async submitValidationResponse(data: ValidationResponse): Promise<{ success: boolean }> {
    try {
      const response = await apiClient.post('/validations/submit', data)
      return { success: true }
    } catch (error) {
      console.error('Error submitting validation response:', error)
      return { success: false }
    }
  },

  async getAdminStats(credentials: any): Promise<any> {
    try {
      const response = await apiClient.post('/admin/stats', credentials)
      return response.data
    } catch (error) {
      console.error('Error fetching admin stats:', error)
      throw error
    }
  }
}

// Mock API Implementation
const mockApi = {
  async reportTranslation(id: string, reason: string): Promise<{ success: boolean }> {
      console.log('Report mock:', id, reason)
      return { success: true }
  },

  async getConversationSample(seenIds: string[] = []): Promise<ConversationSample> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500))
    
    const samples = [
      'రేపు పరీక్ష ఉంది, ఇంకా ఏమీ చదవలేదు.',
      'నిన్న సినిమా చూశావా? చాలా బాగుంది అంటున్నారు.',
      'ఈ వరుసలో నిలబడి ఎంత సేపైంది!',
      'మళ్ళీ బస్సు మిస్ అయింది, ఇప్పుడు ఆఫీసుకి ఎలా వెళ్ళాలి?',
      'కొత్త ఫోన్ కొన్నాను, చాలా ఖరీదైంది కానీ.',
      'A: నీకు ఆ పాట నచ్చిందా?\nB: అవును, చాలా బాగుంది.',
      'రాజు: నువ్వు హోంవర్క్ చేశావా?\nరాధ: ఔను, చాలా సులువుగా ఉంది.',
      'అమ్మ: తిన్నావా?\nపిల్లాడు: ఇంకా ఆకలిగా ఉంది.',
    ]
    
    const randomSample = samples[Math.floor(Math.random() * samples.length)]
    
    return {
      id: `conv_${Date.now()}_${Math.random()}`,
      conversation: randomSample,
      displayCount: Math.floor(Math.random() * 10)
    }
  },

  async getTranslationSample(seenIds: string[] = [], isGold: boolean = false): Promise<TranslationSample> {
    await new Promise(resolve => setTimeout(resolve, 500))
    
    if (isGold) {
        return {
            id: `gold_${Date.now()}`,
            utterance: 'Gold Utterance',
            isGold: true,
            displayCount: 0
        }
    }
    
    const samples = [
      {
        utterance: 'A: నీకు ఆ పాట నచ్చిందా?\nB: చాలా బాగుంది!',
      },
      {
        utterance: 'A: ఈ రోజు వాతావరణం ఎలా ఉంది?\nB: అద్భుతంగా ఉంది, వర్షం పడుతోంది.',
      },
      {
        utterance: 'A: నువ్వు హోంవర్క్ చేశావా?\nB: ఔను, చాలా సులువుగా ఉంది.',
      },
    ]
    
    const randomSample = samples[Math.floor(Math.random() * samples.length)]
    
    return {
      id: `trans_${Date.now()}_${Math.random()}`,
      utterance: randomSample.utterance,
      displayCount: Math.floor(Math.random() * 10)
    }
  },

  async submitConversationResponse(response: ConversationResponse): Promise<{ success: boolean }> {
    await new Promise(resolve => setTimeout(resolve, 500))
    console.log('Submitting conversation response (Mock):', response)
    return { success: true }
  },

  async submitValidationResponse(response: ValidationResponse): Promise<{ success: boolean }> {
    await new Promise(resolve => setTimeout(resolve, 500))
    console.log('Submitting validation response (Mock):', response)
    return { success: true }
  },

  async getAdminStats(credentials: any): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 500))
    if (credentials.username === 'admin' && credentials.password === 'admin123') {
        return {
            totalResponses: 247,
            conversationResponses: 156,
            validationResponses: 91,
            sarcasticCount: 45,
            nonSarcasticCount: 46,
            demographics: {
                regions: { 'Hyderabad': 45, 'Visakhapatnam': 32, 'Guntur': 28, 'Others': 142 },
                ageGroups: { '18-25': 89, '26-35': 76, '36-45': 45, '46+': 37 },
                education: { 'Graduate': 102, 'Post Graduate': 67, 'Other': 78 },
                gender: { 'Male': 132, 'Female': 98, 'Other': 17 }
            }
        }
    }
    throw new Error('Invalid credentials')
  }
}

// Export the appropriate API implementation
export const api = USE_BACKEND ? realApi : mockApi
