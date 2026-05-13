# Telugu Sarcasm Dataset Collection - Frontend

A beautiful, intuitive web interface for collecting Telugu sarcasm data through conversation completion and translation validation.

## Features

### 🌟 Core Functionality
- **Two Collection Methods**:
  1. **Complete Conversations**: Users provide both sarcastic and non-sarcastic responses to conversation starters
  2. **Validate Translations**: Users mark translated conversations as sarcastic or not, and provide examples

### 🎨 User Experience
- **Bilingual Interface**: All text in English with Telugu translations
- **Colorful & Interactive**: Light mode with vibrant gradients and smooth animations
- **Fully Responsive**: Works seamlessly on desktop, tablet, and mobile devices
- **Demographic Collection**: Captures region, age, education, and gender data

### ⌨️ Input Methods
- **Text Input**: Standard typing with auto-transliteration
- **English to Telugu Conversion**: Type in English, press space to convert to Telugu script
- **Virtual Telugu Keyboard**: Full Telugu keyboard with vowels, consonants, matras, and numbers
- **Voice Input**: Record audio responses using browser microphone

### 🎯 Design Highlights
- Smooth animations and transitions
- Color-coded sections with meaningful contrasts
- Interactive hover effects
- Progress tracking
- Loading states
- Skip and quit options

## Tech Stack

- **Framework**: Next.js 14 (React)
- **Language**: TypeScript
- **Styling**: CSS Modules
- **Fonts**: Noto Sans Telugu (Google Fonts)
- **APIs**: Mock API (ready for backend integration)

## Getting Started

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open your browser and visit:
```
http://localhost:3000
```

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── admin/                   # Admin dashboard
│   │   ├── complete-conversation/    # Conversation completion page
│   │   ├── validate-translation/     # Translation validation page
│   │   ├── globals.css              # Global styles
│   │   ├── layout.tsx               # Root layout
│   │   └── page.tsx                 # Home page
│   ├── components/
│   │   ├── DemographicForm.tsx      # Demographics collection form
│   │   └── TeluguKeyboard.tsx       # Virtual Telugu keyboard
│   └── utils/
│       ├── api.ts                   # API functions (mock)
│       ├── constants.ts             # App constants
│       └── transliteration.ts       # English to Telugu conversion
├── package.json
├── tsconfig.json
└── next.config.js
```

## Features in Detail

### Home Page
- Eye-catching introduction
- Motivation section explaining the project
- Two large, interactive option cards for choosing collection method
- Admin dashboard link in footer

### Admin Dashboard
- Secure login (demo: admin/admin123)
- Statistics overview (total responses, sarcastic/non-sarcastic counts)
- Demographic charts (region, age, education, gender distribution)
- Responsive bar and pie charts

### Demographic Form
- District selection (all Telangana & Andhra Pradesh districts)
- Age group selection (radio buttons)
- Education level dropdown
- Gender selection
- Validates all fields before allowing continuation

### Complete Conversations Page
- Displays conversation starter in a highlighted card
- Two-stage process (sarcastic → non-sarcastic)
- Visual stage indicator
- Choice between text and voice input
- Telugu keyboard toggle
- Real-time transliteration
- Skip and submit functionality
- Progress counter

### Validate Translation Page
- Shows full conversation context
- Highlights the response in question
- Binary choice (Sarcastic / Non-Sarcastic)
- Conditional input: If non-sarcastic, asks for a sarcastic example
- Same input methods as conversation page
- Progress tracking

### Telugu Keyboard
- Tabbed interface (Consonants, Vowels, Matras, Numbers)
- All Telugu characters organized logically
- Responsive grid layout
- Space and backspace buttons
- Smooth animations
- Modal overlay with backdrop

### Transliteration
- Automatic English to Telugu conversion
- Triggered by space bar
- Supports common words and phonetic patterns
- Preserves non-transliterable text

## Backend Integration

The frontend is ready for backend integration. Replace the mock API in `src/utils/api.ts` with real API calls:

### Required Endpoints

1. **GET /api/conversation-sample**
   - Returns: `{ id, conversation, displayCount }`

2. **GET /api/translation-sample**
   - Returns: `{ id, conversation, response, displayCount }`

3. **POST /api/conversation-response**
   - Body: `{ sampleId, sarcasticResponse, nonSarcasticResponse, demographics, responseType, audioData? }`

4. **POST /api/validation-response**
   - Body: `{ sampleId, isSarcastic, sarcasticExample?, demographics, responseType?, audioData? }`

### Environment Variables

Create a `.env.local` file:
```
NEXT_PUBLIC_API_URL=http://your-backend-url:8000
```

## Customization

### Colors
Edit CSS variables in `src/app/globals.css`:
```css
:root {
  --primary: #FF6B6B;      /* Main brand color */
  --secondary: #4ECDC4;    /* Secondary brand color */
  --accent: #FFE66D;       /* Accent highlights */
  --success: #95E1D3;      /* Success states */
  --info: #A8E6CF;         /* Info backgrounds */
  /* ... */
}
```

### Districts/Options
Edit arrays in `src/utils/constants.ts`:
- `districts`
- `ageGroups`
- `educationLevels`
- `genders`

### Transliteration Rules
Add mappings in `src/utils/transliteration.ts`:
```typescript
const transliterationMap = {
  'your-word': 'తెలుగు-పదం',
  // ...
}
```

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers (iOS Safari, Chrome Mobile)

**Note**: Voice input requires browser support for `MediaRecorder` API and microphone permissions.

## Build for Production

```bash
npm run build
npm start
```

## Future Enhancements

- [x] Admin dashboard (completed)
- [ ] Real-time statistics updates via WebSocket
- [ ] Analytics integration
- [ ] Enhanced transliteration with ML models
- [ ] Offline support (PWA)
- [ ] User authentication
- [ ] Sample history/editing
- [ ] Export data to CSV/Excel
- [ ] Detailed response viewer

## Contributing

This is a research project for Telugu sarcasm detection. Contributions welcome!

## License

[Your License Here]

## Acknowledgments

- Noto Sans Telugu font by Google Fonts
- Next.js team for the amazing framework
- IIIT Hyderabad BTP project
