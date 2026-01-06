# Frontend Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at: **http://localhost:3000**

### 3. Make Sure Backend is Running

The frontend connects to the backend API on port 8000. Make sure it's running:

```bash
# In another terminal
cd backend
python main.py
```

## Features

✅ **Card Validation** - Real-time format checking (4XXX-****-****-XXXX)  
✅ **User Type Selection** - Student, Traveler, or Family  
✅ **Beautiful Loading** - Animated spinner while fetching benefits  
✅ **Aesthetic Design** - Modern gradient UI with smooth animations  
✅ **Responsive** - Works on desktop, tablet, and mobile  
✅ **Error Handling** - Clear error messages for invalid inputs  

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main page component
│   └── globals.css         # Global styles and animations
├── components/
│   ├── CardInput.tsx           # Card input with validation
│   ├── UserTypeSelector.tsx   # User type selection cards
│   ├── LoadingSpinner.tsx     # Loading animation
│   └── BenefitsDisplay.tsx    # Results display component
├── package.json
├── tailwind.config.js
└── next.config.js
```

## Customization

### Change Port

Edit `package.json`:
```json
"scripts": {
  "dev": "next dev -p 3001"  // Change to any port
}
```

### Change Backend URL

Edit `app/page.tsx` line ~60:
```typescript
const response = await axios.post(
  'http://localhost:8000/benefits',  // Change URL here
  // ...
)
```

## Troubleshooting

### "Cannot connect to server"
- Make sure backend is running on port 8000
- Check: `curl http://localhost:8000/`

### "Module not found"
- Run: `npm install`
- Delete `node_modules` and reinstall if needed

### Port already in use
- Change port in `package.json`
- Or kill process: `lsof -ti:3000 | xargs kill`

## Build for Production

```bash
npm run build
npm start
```

## Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first CSS
- **Axios** - HTTP client for API calls

