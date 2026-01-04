'use client'

import { useState, useEffect, useRef } from 'react'
import CardInput from '@/components/CardInput'
import UserTypeSelector from '@/components/UserTypeSelector'
import LocationSelector from '@/components/LocationSelector'
import LoadingSpinner from '@/components/LoadingSpinner'
import BenefitsDisplay from '@/components/BenefitsDisplay'
import LanguageSwitcher from '@/components/LanguageSwitcher'
import { useLanguage } from '@/contexts/LanguageContext'
import { Language } from '@/lib/translations'
import axios from 'axios'

type UserType = 'student' | 'travel' | 'dining_entertainment_shopping' | 'services' | null

interface BenefitResponse {
  status: string
  card_tier?: string
  recommended_benefit?: {
    explanation: string
    is_beacon_choice?: boolean
    source_chunks: Array<{
      chunk_id: string
      similarity: number
      content?: string
    }>
    scores?: {
      total: number
      lifestyle: number
      location: number
      temporal: number
      monetary: number
    }
  }
  recommendations?: Array<{
    explanation: string
    is_beacon_choice?: boolean
    source_chunks: Array<{
      chunk_id: string
      similarity: number
      content?: string
    }>
    scores?: {
      total: number
      lifestyle: number
      location: number
      temporal: number
      monetary: number
    }
  }>
  all_benefits?: Array<{
    explanation: string
    is_beacon_choice?: boolean
    source_chunks: Array<{
      chunk_id: string
      similarity: number
      content?: string
    }>
    scores?: {
      total: number
      lifestyle: number
      location: number
      temporal: number
      monetary: number
    }
  }>
  total_benefits_count?: number
  disclaimers?: string[]
  language?: string
  metadata?: {
    bin_validated: boolean
    rag_grounded: boolean
    compliance_approved: boolean
  }
  error_code?: string
  message?: string
}

export default function Home() {
  const { language, t } = useLanguage()
  const [cardNumber, setCardNumber] = useState('')
  const [userType, setUserType] = useState<UserType>(null)
  const [location, setLocation] = useState<string | null>(null)
  const [cardType, setCardType] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [benefits, setBenefits] = useState<BenefitResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  // Store last submitted form data to re-fetch when language changes
  const lastSubmittedData = useRef<{
    cardNumber: string
    userType: UserType
    location: string | null
  } | null>(null)
  
  // Track if we should re-fetch on language change (only if benefits are already loaded)
  const shouldRefetchOnLanguageChange = useRef(false)

  // Clear error when user starts interacting
  const handleCardChange = (value: string) => {
    setCardNumber(value)
    if (error && error.includes('card')) {
      setError(null)
    }
  }

  const handleUserTypeChange = (value: UserType) => {
    setUserType(value)
    if (error && !error.includes('card')) {
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!cardNumber || !userType) {
      setError(t('fillRequiredFields'))
      return
    }

    // Validate card format
    const cardPattern = /^4[0-9]{3}-\*{4}-\*{4}-[0-9]{4}$/
    if (!cardPattern.test(cardNumber)) {
      setError(t('invalidCardFormat'))
      return
    }

    setLoading(true)
    setError(null)
    setBenefits(null)

    // Store form data for potential re-fetch on language change
    lastSubmittedData.current = {
      cardNumber,
      userType: userType!,
      location,
    }
    shouldRefetchOnLanguageChange.current = false

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const response = await axios.post<BenefitResponse>(
        `${apiUrl}/benefits`,
        {
          card_number: cardNumber,
          user_context: userType,
          preferred_language: language,
          location: location,
        }
      )

      if (response.data.status === 'success') {
        setBenefits(response.data)
        shouldRefetchOnLanguageChange.current = true
      } else {
        setError(response.data.message || t('failedToFetch'))
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail
      if (typeof errorMessage === 'object' && errorMessage?.message) {
        setError(errorMessage.message)
      } else if (typeof errorMessage === 'string') {
        setError(errorMessage)
      } else {
        setError(t('failedToConnect'))
      }
    } finally {
      setLoading(false)
    }
  }
  
  // Track previous language to detect changes
  const prevLanguageRef = useRef<Language>(language)
  const isInitialMount = useRef(true)
  
  // Re-fetch benefits when language changes (if benefits are already loaded)
  useEffect(() => {
    // Skip on initial mount
    if (isInitialMount.current) {
      isInitialMount.current = false
      prevLanguageRef.current = language
      return
    }
    
    // Only re-fetch if:
    // 1. Language actually changed
    // 2. We have submitted data and should refetch
    // 3. We have the last submitted data
    const languageChanged = prevLanguageRef.current !== language
    const shouldRefetch = shouldRefetchOnLanguageChange.current && lastSubmittedData.current
    
    if (languageChanged && shouldRefetch) {
      const fetchWithNewLanguage = async () => {
        setLoading(true)
        setError(null)
        
        try {
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
          const response = await axios.post<BenefitResponse>(
            `${apiUrl}/benefits`,
            {
              card_number: lastSubmittedData.current!.cardNumber,
              user_context: lastSubmittedData.current!.userType,
              preferred_language: language,
              location: lastSubmittedData.current!.location,
            }
          )

          if (response.data.status === 'success') {
            setBenefits(response.data)
          } else {
            setError(response.data.message || t('failedToFetch'))
          }
        } catch (err: any) {
          const errorMessage = err.response?.data?.detail
          if (typeof errorMessage === 'object' && errorMessage?.message) {
            setError(errorMessage.message)
          } else if (typeof errorMessage === 'string') {
            setError(errorMessage)
          } else {
            setError(t('failedToConnect'))
          }
        } finally {
          setLoading(false)
        }
      }
      
      fetchWithNewLanguage()
    }
    
    // Update previous language reference
    prevLanguageRef.current = language
  }, [language, t]) // Re-run when language changes

  const handleReset = () => {
    setCardNumber('')
    setUserType(null)
    setLocation(null)
    setCardType(null)
    setBenefits(null)
    setError(null)
    lastSubmittedData.current = null
    shouldRefetchOnLanguageChange.current = false
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <LanguageSwitcher />
      <div className="container mx-auto px-4 py-8 md:py-12 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-2xl mb-5 shadow-lg">
            <svg
              className="w-8 h-8 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z"
              />
            </svg>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-indigo-900 via-purple-800 to-indigo-900 bg-clip-text text-transparent mb-3">
            {t('title')}
          </h1>
          <p className="text-lg text-slate-600 max-w-xl mx-auto">
            {t('subtitle')}
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-6 md:p-10">
          {!benefits ? (
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Card Input */}
              <CardInput
                value={cardNumber}
                onChange={handleCardChange}
                onCardTypeDetected={setCardType}
                error={error && error.includes('card') ? error : null}
              />

              {/* User Type Selector */}
              <UserTypeSelector
                value={userType}
                onChange={handleUserTypeChange}
              />

              {/* Location Selector */}
              <div className="relative">
                <LocationSelector
                  value={location}
                  onChange={setLocation}
                />
              </div>

              {/* Error Message */}
              {error && !error.includes('card') && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 animate-fadeIn">
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-red-800 text-sm font-medium flex-1">{error}</p>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || !cardNumber || !userType}
                className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold py-3.5 px-6 rounded-lg shadow-md hover:shadow-lg hover:from-indigo-700 hover:to-purple-700 active:from-indigo-800 active:to-purple-800 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-md flex items-center justify-center gap-2 transform hover:scale-[1.01] active:scale-[0.99]"
              >
                {loading ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span>{t('findingBenefits')}</span>
                  </>
                ) : (
                  <>
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                    <span>{t('findBenefits')}</span>
                  </>
                )}
              </button>
            </form>
          ) : (
            <BenefitsDisplay key={`${benefits.language || 'en'}-${benefits.card_tier || ''}`} benefits={benefits} onReset={handleReset} />
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <div className="inline-flex items-center gap-6 text-xs text-slate-500">
            <div className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
              </svg>
              <span>{t('secure')}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span>{t('verifiedFooter')}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              <span>{t('noDataStored')}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
