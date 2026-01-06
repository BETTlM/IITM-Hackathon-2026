# Visa Benefits Frontend

A beautiful Next.js frontend for the Visa Benefits API.

## Features

- 🎨 Modern, aesthetic UI with gradient designs
- ✅ Real-time card format validation
- 🎯 User type selection (Student/Traveler/Family)
- ⚡ Beautiful loading animations
- 📱 Fully responsive design
- 🔒 Privacy-first (masked cards only)

## Setup

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will run on `http://localhost:3000`

### 3. Make Sure Backend is Running

The frontend expects the backend API to be running on `http://localhost:8000`.

Start the backend:
```bash
cd ../backend
python main.py
```

## Usage

1. Enter your masked Visa card number (format: 4XXX-****-****-XXXX)
2. Select your user type (Student, Traveler, or Family)
3. Click "Find My Benefits"
4. View your personalized benefits!

## Tech Stack

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx      # Root layout
│   ├── page.tsx        # Main page
│   └── globals.css     # Global styles
├── components/
│   ├── CardInput.tsx           # Card input with validation
│   ├── UserTypeSelector.tsx   # User type selection
│   ├── LoadingSpinner.tsx     # Loading animation
│   └── BenefitsDisplay.tsx    # Results display
├── package.json
└── tailwind.config.js
```

## Customization

### Change Port

Edit `package.json`:
```json
{
  "scripts": {
    "dev": "next dev -p 3001"  // Change port here
  }
}
```

### Change Backend URL

Edit `app/page.tsx`:
```typescript
const response = await axios.post(
  'http://localhost:8000/benefits',  // Change URL here
  // ...
)
```

## Build for Production

```bash
npm run build
npm start
```

