'use client'

import { useState } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

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
}

interface BenefitsDisplayProps {
  benefits: BenefitResponse
  onReset: () => void
}

const tierStyles: Record<string, { bg: string; text: string; border: string }> = {
  Classic: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    border: 'border-blue-200',
  },
  Signature: {
    bg: 'bg-purple-50',
    text: 'text-purple-700',
    border: 'border-purple-200',
  },
  Infinite: {
    bg: 'bg-amber-50',
    text: 'text-amber-700',
    border: 'border-amber-200',
  },
}

export default function BenefitsDisplay({ benefits, onReset }: BenefitsDisplayProps) {
  const { t } = useLanguage()
  const [showAll, setShowAll] = useState(false)
  
  const tier = benefits.card_tier || 'Classic'
  const tierStyle = tierStyles[tier] || tierStyles.Classic
  
  const getTierName = (tierName: string): string => {
    if (tierName === 'Classic') return t('classic')
    if (tierName === 'Signature') return t('signature')
    if (tierName === 'Infinite') return t('infinite')
    return tierName
  }
  
  // Get recommendations - show top 4 by default, or all if showAll is true
  const defaultRecommendations = benefits.recommendations || 
    (benefits.recommended_benefit ? [benefits.recommended_benefit] : [])
  
  const allEligibleBenefits = benefits.all_benefits || defaultRecommendations
  const displayRecommendations = showAll ? allEligibleBenefits : defaultRecommendations
  const hasMore = allEligibleBenefits.length > defaultRecommendations.length

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className={`w-12 h-12 ${tierStyle.bg} ${tierStyle.border} border rounded-lg flex items-center justify-center`}>
            <svg
              className={`w-6 h-6 ${tierStyle.text}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">{t('yourBenefits')}</h2>
            <div className="flex items-center gap-2 mt-1">
              <span className={`px-2.5 py-0.5 rounded-md text-xs font-semibold ${tierStyle.bg} ${tierStyle.text} border ${tierStyle.border}`}>
                {getTierName(tier)}
              </span>
              {benefits.metadata?.rag_grounded && (
                <span className="px-2.5 py-0.5 rounded-md text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                  {t('verified')}
                </span>
              )}
              {benefits.total_benefits_count && benefits.total_benefits_count > 0 && (
                <span className="px-2.5 py-0.5 rounded-md text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">
                  {benefits.total_benefits_count} {t('eligible')}
                </span>
              )}
            </div>
          </div>
        </div>
        <button
          onClick={onReset}
          className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-lg transition-all duration-200 active:scale-95"
          title={t('searchAgain')}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* Recommendations */}
      {displayRecommendations.length > 0 && (
        <div className="space-y-3">
          {displayRecommendations.map((rec, idx) => {
            const isBeaconChoice = rec.is_beacon_choice
            
            return (
            <div
              key={idx}
              className={`rounded-lg border-2 ${
                isBeaconChoice
                  ? 'bg-gradient-to-br from-amber-50 to-orange-50 border-amber-300 shadow-sm p-5'
                  : 'bg-white border-slate-200 p-3'
              }`}
            >
              {/* Badge */}
              <div className={`flex items-center gap-2 ${isBeaconChoice ? 'mb-3' : 'mb-2'}`}>
                {isBeaconChoice ? (
                  <span className="px-3 py-1 bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-md text-xs font-bold flex items-center gap-1.5 shadow-sm">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    {t('beaconChoice')}
                  </span>
                ) : (
                  <span className="px-2 py-0.5 bg-slate-100 text-slate-700 rounded text-xs font-medium">
                    {t('option')} {idx + 1}
                  </span>
                )}
                {rec.scores && (
                  <span className={`text-xs ${isBeaconChoice ? 'text-slate-500' : 'text-slate-400'}`}>
                    {(rec.scores.total * 100).toFixed(0)}% {t('match')}
                  </span>
                )}
              </div>

              {/* Explanation - Full for Beacon's Choice, Compact for others */}
              {isBeaconChoice ? (
              <div className="text-slate-700 leading-relaxed mb-4 space-y-2">
                {rec.explanation.split('\n').map((line, idx) => {
                  if (!line.trim()) return <br key={idx} />
                  
                  const trimmedLine = line.trim()
                  let contentLine = trimmedLine
                  let isBullet = false
                  let isNumbered = false
                  let listNumber = ''
                  
                  // Check if line starts with bullet point
                  const bulletMatch = trimmedLine.match(/^[-*•]\s+(.+)$/)
                  if (bulletMatch) {
                    isBullet = true
                    contentLine = bulletMatch[1] // Remove bullet marker
                  }
                  
                  // Check if line starts with numbered list
                  const numberedMatch = trimmedLine.match(/^(\d+)\.\s+(.+)$/)
                  if (numberedMatch) {
                    isNumbered = true
                    listNumber = numberedMatch[1]
                    contentLine = numberedMatch[2] // Remove number marker
                  }
                  
                  // Handle bold text (**text**)
                  const parts: (string | JSX.Element)[] = []
                  let lastIndex = 0
                  const boldRegex = /\*\*(.*?)\*\*/g
                  let match
                  
                  while ((match = boldRegex.exec(contentLine)) !== null) {
                    // Add text before the match
                    if (match.index > lastIndex) {
                      parts.push(contentLine.substring(lastIndex, match.index))
                    }
                    // Add bold text
                    parts.push(
                      <strong key={`bold-${idx}-${match.index}`} className="font-semibold text-slate-900">
                        {match[1]}
                      </strong>
                    )
                    lastIndex = match.index + match[0].length
                  }
                  // Add remaining text
                  if (lastIndex < contentLine.length) {
                    parts.push(contentLine.substring(lastIndex))
                  }
                  
                  // If no bold text found, use the line as-is
                  if (parts.length === 0) {
                    parts.push(contentLine)
                  }
                  
                  // Render based on type
                  if (isNumbered) {
                    return (
                      <div key={idx} className="mt-2 flex items-start">
                        <span className="mr-2 font-semibold text-slate-900">{listNumber}.</span>
                        <span>{parts}</span>
                      </div>
                    )
                  }
                  
                  if (isBullet) {
                    return (
                      <div key={idx} className="mt-1 ml-4 flex items-start">
                        <span className="mr-2">•</span>
                        <span>{parts}</span>
                      </div>
                    )
                  }
                  
                  return (
                    <div key={idx} className={idx > 0 ? 'mt-2' : ''}>
                      {parts}
                    </div>
                  )
                })}
              </div>
              ) : (
              // Compact version for follow-up suggestions
              <div className="text-sm text-slate-600 leading-snug">
                {(() => {
                  const firstLine = rec.explanation.split('\n')[0].replace(/\*\*(.*?)\*\*/g, '$1')
                  return firstLine.length > 150 ? firstLine.substring(0, 150) + '...' : firstLine
                })()}
              </div>
              )}

              {/* Sources - Only show for Beacon's Choice */}
              {isBeaconChoice && rec.source_chunks && rec.source_chunks.length > 0 && (
                <div className="pt-4 border-t border-slate-200">
                  <h5 className="text-xs font-semibold text-slate-600 mb-2 flex items-center gap-1.5">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    {t('sources')}
                  </h5>
                  <div className="space-y-2">
                    {rec.source_chunks.map((chunk, chunkIdx) => (
                      <div key={chunkIdx} className="bg-slate-50 rounded-md p-2.5 border border-slate-200">
                        <div className="flex items-start justify-between mb-1">
                          <span className="text-xs font-medium text-slate-700">
                            {t('source')} {chunkIdx + 1}
                          </span>
                          <span className="text-xs text-slate-500">
                            {Math.min(100, (chunk.similarity * 100)).toFixed(1)}%
                          </span>
                        </div>
                        {chunk.content && (
                          <p className="text-xs text-slate-600 mt-1 line-clamp-2">
                            {chunk.content}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
            )
          })}
        </div>
      )}

      {/* Show All / Show Less Button */}
      {hasMore && (
        <div className="flex justify-center pt-2">
          <button
            onClick={() => setShowAll(!showAll)}
            className="px-4 py-2 text-sm font-medium text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 rounded-lg transition-all duration-200 flex items-center gap-2 active:scale-95"
          >
            {showAll ? (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                </svg>
                <span>{t('showTop4')}</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                <span>{t('showAll')} {benefits.total_benefits_count || allEligibleBenefits.length} {t('eligibleBenefits')}</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Disclaimers */}
      {benefits.disclaimers && benefits.disclaimers.length > 0 && (
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-slate-700 mb-2 flex items-center gap-1.5">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            {t('importantInfo')}
          </h4>
          <ul className="space-y-1.5">
            {benefits.disclaimers.map((disclaimer, idx) => (
              <li key={idx} className="text-xs text-slate-600 flex items-start gap-2">
                <span className="text-slate-400 mt-0.5">•</span>
                <span>{disclaimer}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Action button */}
      <button
        onClick={onReset}
        className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:shadow-lg hover:from-indigo-700 hover:to-purple-700 transition-all duration-200 flex items-center justify-center gap-2 transform hover:scale-[1.01] active:scale-[0.99]"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        <span>{t('searchAgain')}</span>
      </button>
    </div>
  )
}
