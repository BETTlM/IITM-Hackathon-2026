'use client'

import { useState, useEffect } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

interface CardInputProps {
  value: string
  onChange: (value: string) => void
  onCardTypeDetected?: (cardType: string | null) => void
  error?: string | null
}

// BIN to Card Type mapping (matches backend)
const getCardType = (cardNumber: string): string | null => {
  if (!cardNumber || cardNumber.length < 4) return null
  
  const first4 = cardNumber.substring(0, 4)
  
  const binTierMap: Record<string, string> = {
    '4000': 'Signature',
    '4111': 'Classic',
    '4222': 'Infinite',
    '4333': 'Signature',
    '4444': 'Infinite',
  }
  
  return binTierMap[first4] || null
}

const cardTypeStyles: Record<string, { bg: string; border: string; text: string }> = {
  Classic: {
    bg: 'bg-blue-50',
    border: 'border-blue-200',
    text: 'text-blue-700',
  },
  Signature: {
    bg: 'bg-purple-50',
    border: 'border-purple-200',
    text: 'text-purple-700',
  },
  Infinite: {
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    text: 'text-amber-700',
  },
}

export default function CardInput({ value, onChange, onCardTypeDetected, error }: CardInputProps) {
  const { t } = useLanguage()
  const [displayValue, setDisplayValue] = useState('')
  const [cardType, setCardType] = useState<string | null>(null)

  useEffect(() => {
    setDisplayValue(value)
    const detectedType = getCardType(value)
    setCardType(detectedType)
    onCardTypeDetected?.(detectedType)
  }, [value, onCardTypeDetected])

  const formatCardNumber = (input: string) => {
    // Remove all non-digits and dashes
    const digitsOnly = input.replace(/[^\d]/g, '')
    
    // Limit to 16 digits max
    const limitedDigits = digitsOnly.slice(0, 16)
    
    // If user has entered at least 4 digits, auto-fill middle with asterisks
    let result = ''
    if (limitedDigits.length <= 4) {
      // Just first 4 digits - no auto-fill yet
      result = limitedDigits
    } else {
      // First 4 digits + middle 8 asterisks + last 4 digits (being typed)
      const first4 = limitedDigits.substring(0, 4)
      const lastDigits = limitedDigits.substring(4) // This will be 1-4 digits
      const middleAsterisks = '********'
      result = first4 + middleAsterisks + lastDigits
    }

    // Format with dashes
    let formatted = ''
    for (let i = 0; i < result.length; i++) {
      if (i > 0 && i % 4 === 0) {
        formatted += '-'
      }
      formatted += result[i]
    }

    return formatted
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target.value
    const formatted = formatCardNumber(input)
    setDisplayValue(formatted)
    onChange(formatted)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (
      [8, 9, 27, 13, 46].indexOf(e.keyCode) !== -1 ||
      (e.keyCode === 65 && e.ctrlKey === true) ||
      (e.keyCode === 67 && e.ctrlKey === true) ||
      (e.keyCode === 86 && e.ctrlKey === true) ||
      (e.keyCode === 88 && e.ctrlKey === true) ||
      (e.keyCode >= 35 && e.keyCode <= 39)
    ) {
      return
    }
    if (
      (e.shiftKey || (e.keyCode < 48 || e.keyCode > 57)) &&
      (e.keyCode < 96 || e.keyCode > 105) &&
      e.keyCode !== 56
    ) {
      e.preventDefault()
    }
  }

  const isValid = value.match(/^4[0-9]{3}-\*{4}-\*{4}-[0-9]{4}$/)

  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-slate-700 mb-2">
        {t('cardNumber')}
      </label>
      <div className="relative">
        <input
          type="text"
          value={displayValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder="4111-****-****-1111"
          maxLength={19}
          className={`w-full px-4 py-3 text-base bg-white text-slate-900 border-2 rounded-lg focus:outline-none focus:ring-2 transition-all ${
            error
              ? 'border-red-300 focus:border-red-500 focus:ring-red-200'
              : isValid
              ? 'border-emerald-400 focus:border-emerald-500 focus:ring-emerald-200'
              : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-200'
          }`}
        />
        {isValid && !error && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <svg
              className="w-5 h-5 text-emerald-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
        )}
      </div>
      
      {/* Card Type Display */}
      {cardType && isValid && !error && (
        <div className={`${cardTypeStyles[cardType].bg} ${cardTypeStyles[cardType].border} border rounded-lg p-3 flex items-center justify-between animate-fadeIn`}>
          <div className="flex items-center gap-2">
            <span className={`text-sm font-semibold ${cardTypeStyles[cardType].text}`}>
              {cardType === 'Classic' ? t('classic') : cardType === 'Signature' ? t('signature') : cardType === 'Infinite' ? t('infinite') : cardType} {t('card')}
            </span>
          </div>
          <span className={`text-xs font-medium ${cardTypeStyles[cardType].text} px-2 py-1 rounded ${cardTypeStyles[cardType].bg}`}>
            {t('cardDetected')}
          </span>
        </div>
      )}
      
      {error && (
        <div className="flex items-start gap-2 mt-1 animate-fadeIn">
          <svg className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}
      {!error && !cardType && (
        <p className="text-xs text-slate-500 mt-1">
          {t('cardFormat')}
        </p>
      )}
    </div>
  )
}
