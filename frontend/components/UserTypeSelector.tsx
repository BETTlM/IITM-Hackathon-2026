'use client'

import { useLanguage } from '@/contexts/LanguageContext'
import { TranslationKey } from '@/lib/translations'

type UserType = 'student' | 'travel' | 'dining_entertainment_shopping' | 'services' | null

interface UserTypeSelectorProps {
  value: UserType
  onChange: (value: UserType) => void
}

const getUserTypes = (t: (key: TranslationKey) => string) => [
  {
    id: 'student' as const,
    label: t('student'),
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    ),
    description: t('studentDesc'),
  },
  {
    id: 'travel' as const,
    label: t('travel'),
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    description: t('travelDesc'),
  },
  {
    id: 'dining_entertainment_shopping' as const,
    label: t('dining'),
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
      </svg>
    ),
    description: t('diningDesc'),
  },
  {
    id: 'services' as const,
    label: t('services'),
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    description: t('servicesDesc'),
  },
]

export default function UserTypeSelector({ value, onChange }: UserTypeSelectorProps) {
  const { t } = useLanguage()
  const userTypes = getUserTypes(t)
  
  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-slate-700 mb-3">
        {t('userTypeLabel')}
      </label>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {userTypes.map((type) => (
          <button
            key={type.id}
            type="button"
            onClick={() => onChange(type.id)}
            className={`relative p-4 rounded-lg border-2 transition-all duration-200 text-left ${
              value === type.id
                ? type.id === 'student'
                  ? 'border-blue-600 bg-blue-600 text-white shadow-md scale-[1.02]'
                  : type.id === 'travel'
                  ? 'border-purple-600 bg-purple-600 text-white shadow-md scale-[1.02]'
                  : type.id === 'dining_entertainment_shopping'
                  ? 'border-emerald-600 bg-emerald-600 text-white shadow-md scale-[1.02]'
                  : 'border-amber-600 bg-amber-600 text-white shadow-md scale-[1.02]'
                : 'border-slate-200 bg-white text-slate-900 hover:border-slate-300 hover:bg-slate-50 hover:shadow-sm active:scale-[0.98]'
            }`}
          >
            <div className="flex items-start gap-3">
              <div className={`flex-shrink-0 ${value === type.id ? 'text-white' : 'text-slate-400'}`}>
                {type.icon}
              </div>
              <div className="flex-1 min-w-0">
                <h3 className={`font-semibold text-base mb-0.5 ${value === type.id ? 'text-white' : 'text-slate-900'}`}>
                  {type.label}
                </h3>
                <p className={`text-xs ${value === type.id ? 'text-white/80' : 'text-slate-600'}`}>
                  {type.description}
                </p>
              </div>
              {value === type.id && (
                <div className="flex-shrink-0">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
