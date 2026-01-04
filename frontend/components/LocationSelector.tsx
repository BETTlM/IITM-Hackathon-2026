'use client'

import { useState, useEffect, useRef } from 'react'
import { useLanguage } from '@/contexts/LanguageContext'

interface LocationSelectorProps {
  value: string | null
  onChange: (value: string | null) => void
}

const majorCities = [
  'Chennai',
  'Mumbai',
  'Goa',
  'Bangalore',
]

const importantPlaces = {
  Chennai: [
    'Marina Beach',
    'T. Nagar',
    'Anna Nagar',
    'Velachery',
    'OMR (IT Corridor)',
    'Phoenix Mall',
    'Express Avenue',
  ],
  Mumbai: [
    'Marine Drive',
    'Bandra',
    'Andheri',
    'Powai',
    'Lower Parel',
    'Phoenix Mills',
    'Juhu Beach',
  ],
  Goa: [
    'Calangute Beach',
    'Baga Beach',
    'Anjuna',
    'Panjim',
    'Vasco da Gama',
    'Margao',
    'Candolim',
  ],
  Bangalore: [
    'MG Road',
    'Indiranagar',
    'Koramangala',
    'Whitefield',
    'Electronic City',
    'Phoenix Marketcity',
    'UB City',
  ],
}

// Advanced fuzzy search - Google Maps style
const fuzzySearch = (query: string): Array<{ city: string; place: string | null; score: number }> => {
  if (!query || query.length < 1) return []
  
  const results: Array<{ city: string; place: string | null; score: number }> = []
  const lowerQuery = query.toLowerCase().trim()
  
  // Remove dots and normalize
  const normalizedQuery = lowerQuery.replace(/\./g, '').replace(/\s+/g, ' ')
  
  // Search cities
  for (const city of majorCities) {
    const lowerCity = city.toLowerCase()
    
    // Exact match
    if (lowerCity === lowerQuery) {
      results.push({ city, place: null, score: 100 })
      continue
    }
    
    // Starts with
    if (lowerCity.startsWith(lowerQuery)) {
      results.push({ city, place: null, score: 80 })
      continue
    }
    
    // Contains
    if (lowerCity.includes(lowerQuery)) {
      results.push({ city, place: null, score: 60 })
      continue
    }
    
    // Fuzzy match (handles typos, abbreviations)
    const cityWords = lowerCity.split(' ')
    const queryWords = normalizedQuery.split(' ')
    
    // Check if query words match city words (handles "T.n.." -> "T. Nagar")
    let fuzzyScore = 0
    let matchedWords = 0
    
    for (const qWord of queryWords) {
      for (const cWord of cityWords) {
        // Remove dots and compare
        const cleanQWord = qWord.replace(/\./g, '')
        const cleanCWord = cWord.replace(/\./g, '').replace(/\s+/g, '')
        
        if (cleanCWord.startsWith(cleanQWord) || cleanCWord.includes(cleanQWord)) {
          fuzzyScore += 30
          matchedWords++
          break
        }
      }
    }
    
    if (matchedWords > 0) {
      results.push({ city, place: null, score: fuzzyScore })
    }
  }
  
  // Search places within cities
  for (const [city, places] of Object.entries(importantPlaces)) {
    for (const place of places) {
      const lowerPlace = place.toLowerCase()
      const normalizedPlace = lowerPlace.replace(/\./g, '').replace(/\s+/g, ' ')
      
      // Exact match
      if (normalizedPlace === normalizedQuery || lowerPlace === lowerQuery) {
        results.push({ city, place, score: 95 })
        continue
      }
      
      // Starts with
      if (normalizedPlace.startsWith(normalizedQuery) || lowerPlace.startsWith(lowerQuery)) {
        results.push({ city, place, score: 85 })
        continue
      }
      
      // Contains
      if (normalizedPlace.includes(normalizedQuery) || lowerPlace.includes(lowerQuery)) {
        results.push({ city, place, score: 70 })
        continue
      }
      
      // Fuzzy match for places (handles "T.n.." -> "T. Nagar")
      const placeWords = normalizedPlace.split(' ')
      const queryWords = normalizedQuery.split(' ')
      
      let fuzzyScore = 0
      let matchedWords = 0
      
      for (const qWord of queryWords) {
        for (const pWord of placeWords) {
          const cleanQWord = qWord.replace(/\./g, '')
          const cleanPWord = pWord.replace(/\./g, '').replace(/\s+/g, '')
          
          // Handle abbreviations like "T.n.." -> "T Nagar"
          if (cleanPWord.startsWith(cleanQWord) || cleanPWord.includes(cleanQWord)) {
            fuzzyScore += 25
            matchedWords++
            break
          }
          
          // Handle reverse - if query is abbreviation
          if (cleanQWord.length >= 2 && cleanPWord.startsWith(cleanQWord.substring(0, 2))) {
            fuzzyScore += 20
            matchedWords++
            break
          }
        }
      }
      
      if (matchedWords > 0) {
        results.push({ city, place, score: fuzzyScore })
      }
    }
  }
  
  // Remove duplicates and sort by score
  const uniqueResults = new Map<string, { city: string; place: string | null; score: number }>()
  
  for (const result of results) {
    const key = `${result.city}-${result.place || ''}`
    const existing = uniqueResults.get(key)
    if (!existing || result.score > existing.score) {
      uniqueResults.set(key, result)
    }
  }
  
  return Array.from(uniqueResults.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, 10) // Top 10 results
}

export default function LocationSelector({ value, onChange }: LocationSelectorProps) {
  const { t, language } = useLanguage()
  const [searchQuery, setSearchQuery] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const [isGettingLocation, setIsGettingLocation] = useState(false)
  const [locationError, setLocationError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Use fuzzy search instead of simple filter
  const searchResults = searchQuery ? fuzzySearch(searchQuery) : []

  const selectedCityPlaces = value ? importantPlaces[value as keyof typeof importantPlaces] : []

  // Track if we're in "selection mode" vs "search mode"
  const [isSelecting, setIsSelecting] = useState(false)
  const lastSelectedDisplayRef = useRef<string>('')

  // Sync searchQuery with selected value when value changes externally
  useEffect(() => {
    if (value && !isSelecting) {
      // If we have a last selected display value that matches the city, use it
      // Otherwise, just use the city name
      if (lastSelectedDisplayRef.current && lastSelectedDisplayRef.current.includes(value)) {
        setSearchQuery(lastSelectedDisplayRef.current)
      } else {
        setSearchQuery(value)
      }
    } else if (!value && !showDropdown && !isSelecting) {
      // Clear input when value is cleared and dropdown is closed
      setSearchQuery('')
      lastSelectedDisplayRef.current = ''
    }
  }, [value, isSelecting, showDropdown])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false)
        // If a location is selected, restore it in the input
        if (value) {
          setSearchQuery(value)
        } else {
          setSearchQuery('')
        }
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [value])

  const handleGetLocation = () => {
    setIsGettingLocation(true)
    setLocationError(null)

    if (!navigator.geolocation) {
      setLocationError(t('geolocationNotSupported'))
      setIsGettingLocation(false)
      return
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const response = await fetch(
            `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${position.coords.latitude}&longitude=${position.coords.longitude}&localityLanguage=en`
          )
          const data = await response.json()
          const city = data.city || data.locality || 'Unknown'

          const matchedCity = majorCities.find(
            (c) => c.toLowerCase() === city.toLowerCase()
          )

          if (matchedCity) {
            lastSelectedDisplayRef.current = matchedCity
            onChange(matchedCity)
            setSearchQuery(matchedCity)
          } else {
            const closestMatch = majorCities.find((c) =>
              c.toLowerCase().includes(city.toLowerCase().split(' ')[0])
            )
            if (closestMatch) {
              lastSelectedDisplayRef.current = closestMatch
              onChange(closestMatch)
              setSearchQuery(closestMatch)
            } else {
              setLocationError(`${t('detectedLocation')} ${city}. ${t('selectFromList')}`)
            }
          }
        } catch (err) {
          setLocationError(t('failedToGetLocation'))
        } finally {
          setIsGettingLocation(false)
        }
      },
      (error) => {
        setLocationError(t('unableToGetLocation'))
        setIsGettingLocation(false)
      }
    )
  }

  const handleSelect = (city: string, place: string | null = null) => {
    setIsSelecting(true)
    const displayValue = place ? `${place}, ${city}` : city
    lastSelectedDisplayRef.current = displayValue
    onChange(city)
    setSearchQuery(displayValue)
    setShowDropdown(false)
    setLocationError(null)
    // Reset selecting flag after a brief delay
    setTimeout(() => setIsSelecting(false), 100)
  }

  const handleClear = () => {
    setIsSelecting(true)
    onChange(null)
    setSearchQuery('')
    lastSelectedDisplayRef.current = ''
    setLocationError(null)
    setTimeout(() => {
      setIsSelecting(false)
      inputRef.current?.focus()
    }, 100)
  }

  const handleInputFocus = () => {
    setShowDropdown(true)
    // If there's a selected value and input matches it, allow editing
    if (value && (searchQuery === value || searchQuery.includes(value))) {
      // User wants to change location - keep current input but allow editing
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value
    setSearchQuery(newValue)
    setShowDropdown(true)
    setIsSelecting(false)
    // If user is typing and there was a selection, clear it if they're changing it significantly
    if (value && newValue !== value && !newValue.toLowerCase().includes(value.toLowerCase())) {
      onChange(null)
    }
  }

  return (
    <div className="space-y-2 relative">
      <label className="block text-sm font-semibold text-slate-700 mb-2">
        {t('location')} <span className="text-slate-400 font-normal">{t('locationOptional')}</span>
      </label>

      {/* Search Input */}
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={searchQuery}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          placeholder={t('locationPlaceholder')}
          className={`w-full px-4 py-3 text-base bg-white text-slate-900 border-2 rounded-lg focus:outline-none focus:ring-2 transition-all ${
            value
              ? 'border-indigo-300 focus:border-indigo-500 focus:ring-indigo-200'
              : 'border-slate-300 focus:border-indigo-500 focus:ring-indigo-200'
          }`}
        />
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-12 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors p-1 rounded hover:bg-slate-100 active:scale-95"
            title="Clear location"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
        <button
          type="button"
          onClick={handleGetLocation}
          disabled={isGettingLocation}
          className="absolute right-2 top-1/2 -translate-y-1/2 px-3 py-1.5 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-1.5 text-sm shadow-sm hover:shadow-md active:scale-95"
          title={t('getLocation')}
        >
          {isGettingLocation ? (
            <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          )}
        </button>
      </div>

      {/* Dropdown with fuzzy search results */}
      {showDropdown && searchQuery && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-white border border-slate-200 rounded-lg shadow-lg max-h-64 overflow-y-auto top-full"
        >
          {searchResults.length > 0 ? (
            <ul className="py-1">
              {searchResults.map((result, idx) => (
                <li key={`${result.city}-${result.place || 'city'}-${idx}`}>
                  <button
                    type="button"
                    onClick={() => handleSelect(result.city, result.place || null)}
                    className="w-full px-4 py-2.5 text-left hover:bg-indigo-50 active:bg-indigo-100 transition-colors flex items-start gap-3 text-sm group"
                  >
                    <svg className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <div className="flex-1 min-w-0">
                      {result.place ? (
                        <>
                          <div className="font-medium text-slate-900 group-hover:text-indigo-700">
                            {result.place}
                          </div>
                          <div className="text-xs text-slate-500 mt-0.5">
                            {result.city}
                          </div>
                        </>
                      ) : (
                        <div className="font-medium text-slate-900 group-hover:text-indigo-700">
                          {result.city}
                        </div>
                      )}
                    </div>
                    {result.score >= 80 && (
                      <span className="text-xs text-emerald-600 font-medium flex-shrink-0">
                        {t('bestMatch')}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="px-4 py-3 text-slate-500 text-sm text-center">
              {t('noResults')}
            </div>
          )}
        </div>
      )}

      {/* Selected City Display - Show additional info below input */}
      {value && (searchQuery === value || searchQuery === lastSelectedDisplayRef.current || (lastSelectedDisplayRef.current && searchQuery.includes(value))) && !showDropdown && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 animate-fadeIn">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-indigo-600 rounded-md flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div className="min-w-0">
                <p className="font-semibold text-slate-900 text-sm">{value}</p>
                {selectedCityPlaces && selectedCityPlaces.length > 0 && (
                  <p className="text-xs text-slate-600 mt-0.5 truncate">
                    {t('popularAreas')} {selectedCityPlaces.slice(0, 3).join(', ')}
                  </p>
                )}
              </div>
            </div>
            <button
              type="button"
              onClick={handleClear}
              className="text-slate-400 hover:text-slate-600 hover:bg-indigo-100 rounded p-1 transition-colors flex-shrink-0"
              title="Clear location"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {locationError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 animate-fadeIn">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-red-800 flex-1">{locationError}</p>
          </div>
        </div>
      )}
    </div>
  )
}
