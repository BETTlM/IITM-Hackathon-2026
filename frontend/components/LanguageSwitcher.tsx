'use client'

import { useLanguage } from '@/contexts/LanguageContext'

export default function LanguageSwitcher() {
  const { language, setLanguage } = useLanguage()

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'ta' : 'en')
  }

  return (
    <div className="fixed top-4 right-4 z-50 flex items-center gap-3">
      <span className={`text-sm font-medium transition-colors ${language === 'en' ? 'text-slate-900' : 'text-slate-400'}`}>
        EN
      </span>
      <button
        onClick={toggleLanguage}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
          language === 'ta' ? 'bg-indigo-600' : 'bg-slate-300'
        }`}
        role="switch"
        aria-checked={language === 'ta'}
        aria-label="Toggle language"
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
            language === 'ta' ? 'translate-x-6' : 'translate-x-1'
          }`}
        />
      </button>
      <span className={`text-sm font-medium transition-colors ${language === 'ta' ? 'text-slate-900' : 'text-slate-400'}`}>
        தமிழ்
      </span>
    </div>
  )
}

