# 🌱 AgriTech Assistant

A comprehensive mobile application for smart farming solutions powered by AI. This React Native app, built with Expo, connects to a FastAPI backend to provide crop recommendations, soil type detection, and water scarcity forecasting.

## 📱 Features

### 🌾 Crop Prediction
- AI-powered crop recommendations based on soil nutrients (N, P, K, pH)
- Location-based weather analysis
- Soil type integration (manual selection or image detection)
- Top 5 crop suggestions with confidence scores
- Water requirement categorization

### 🏞️ Soil Type Detection
- Image-based soil classification using CNN
- Support for 5 soil types: Black, Cinder, Laterite, Peat, Yellow
- Real-time image analysis with confidence scores
- Camera and gallery integration

### 💧 Water Scarcity Forecast
- Multi-month water availability predictions
- Risk level assessment (High/Medium/Low)
- Drought-resistant crop recommendations
- Water conservation strategies
- Weather forecast summaries

### 🎨 Additional Features
- Light/Dark theme support with system theme detection
- Comprehensive crops database with water requirements
- Backend health monitoring
- Professional UI with Material Design 3
- Offline-ready architecture

## 🚀 Getting Started

### Prerequisites
- **Node.js**: 18.x or higher
- **Python**: 3.8 - 3.11
- **Git**: Latest version for version control
- **Expo Go**: App on your mobile device for testing
- **Android Studio**: For Android development
- **Xcode**: For iOS development (macOS only)

### Quick Start

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd CropRecomendation
   ```

2. **Frontend Setup (React Native + Expo)**
   ```bash
   # Navigate to frontend directory
   cd frontend

   # Install dependencies
   npm install

   # Install Expo CLI globally (if not installed)
   npm install -g @expo/cli

   # Start the development server
   npx expo start --tunnel
   ```

3. **Backend Setup (Python FastAPI)**
   ```bash
   # Navigate to backend directory
   cd backend

   # Create virtual environment
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

   # Install Python dependencies
   pip install -r ../requirements.txt

   # Start the backend server
   python main.py
   ```

4. **Mobile Testing**
   - Install **Expo Go** from App Store/Play Store
   - Scan the QR code from the Expo development server
   - The app will load on your mobile device

## 🏗️ Project Structure

```
CropRecomendation/
├── backend/
│   ├── main.py                    # FastAPI backend entry point
│   └── model_artifacts/          # Machine learning model files
│       ├── best_soil_cnn.pth     # Soil type CNN model
│       ├── feature_medians.joblib # Feature medians for preprocessing
│       ├── feature_names.joblib   # Feature names for model
│       ├── label_encoder.joblib   # Label encoder for output classes
│       ├── model.joblib          # Crop prediction model
│       └── scaler.joblib         # Feature scaler
├── frontend/
│   ├── android/                  # Android-specific configurations
│   │   └── app/src/main/
│   │       └── AndroidManifest.xml
│   ├── ios/                      # iOS-specific configurations
│   │   └── AgriTechApp/
│   │       └── Info.plist
│   ├── assets/                   # Static assets for the app
│   │   ├── adaptive-icon.png
│   │   ├── favicon.png
│   │   ├── icon.png
│   │   └── splash.png
│   ├── App.tsx                   # Root React Native component
│   ├── app.json                  # Expo configuration
│   ├── babel.config.js           # Babel configuration
│   ├── index.js                  # Entry point for React Native
│   ├── package.json              # Frontend dependencies
│   ├── package-lock.json         # Dependency lock file
│   └── tsconfig.json             # TypeScript configuration
├── training-file/                # Training notebooks
│   ├── crop-recomendation.ipynb  # Crop prediction model training
│   └── soil-classification.ipynb # Soil classification model training
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies for backend
```

## 🔧 Tech Stack Requirements

### Frontend (React Native with Expo)

#### Core Dependencies
```json
{
  "expo": "~51.0.28",
  "react": "18.2.0",
  "react-native": "0.74.5",
  "@expo/vector-icons": "^14.0.2",
  "expo-image-picker": "~15.1.0",
  "expo-location": "~17.0.1",
  "expo-secure-store": "~13.0.2",
  "expo-status-bar": "~1.12.1",
  "react-native-paper": "^5.12.3",
  "react-native-vector-icons": "^10.1.0",
  "axios": "^1.7.7",
  "@react-navigation/native": "6.1.7",
  "@react-navigation/bottom-tabs": "6.5.8",
  "@react-navigation/stack": "6.3.17",
  "react-native-screens": "3.22.1",
  "react-native-safe-area-context": "4.7.1",
  "react-native-gesture-handler": "2.12.1",
  "react-native-svg": "13.10.0",
  "react-native-animatable": "1.3.3",
  "@react-native-async-storage/async-storage": "1.19.1",
  "react-native-image-picker": "5.6.0",
  "react-native-geolocation-service": "5.3.1",
  "react-native-permissions": "3.8.4",
  "@react-native-picker/picker": "2.4.10",
  "react-native-linear-gradient": "2.8.1",
  "react-native-modal": "13.0.1"
}
```

#### Development Dependencies
```json
{
  "@babel/core": "^7.20.0",
  "@types/react": "~18.2.45",
  "@types/react-native": "^0.73.0",
  "typescript": "~5.3.3"
}
```

#### System Requirements
- **Node.js**: 18.x or higher
- **npm**: 9.x or higher
- **Expo CLI**: Latest version
- **Android Emulator**: API level 21+ (Android 5.0+)
- **iOS Simulator**: iOS 13.4+

### Backend (Python FastAPI)

#### Core Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
torch==2.1.1
torchvision==0.16.1
Pillow==10.1.0
python-multipart==0.0.6
joblib==1.3.2
prophet==1.1.5
statsmodels==0.14.0
xgboost==2.0.2
```

#### Optional ML Models
- Crop prediction model (`model.joblib`)
- Soil type CNN model (`best_soil_cnn.pth`)
- Feature preprocessing (`feature_medians.joblib`, `feature_names.joblib`, `label_encoder.joblib`, `scaler.joblib`)

#### System Requirements
- **Python**: 3.8 - 3.11
- **pip**: Latest version
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies

## 🔧 Configuration

### Backend Connection
Update the server URL in `frontend/src/services/api.ts`:
```typescript
const BASE_URL = 'http://localhost:8000'; // Development
// For Android emulator
// const BASE_URL = 'http://10.0.2.2:8000';
// For iOS simulator
// const BASE_URL = 'http://localhost:8000';
// For production
// const BASE_URL = 'https://your-api-domain.com';
```

### Environment Variables
Create a `.env` file in the `frontend/` directory:
```env
# Development
API_BASE_URL=http://localhost:8000
EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0
REACT_NATIVE_PACKAGER_HOSTNAME=localhost

# Production
# API_BASE_URL=https://your-api-domain.com
# SENTRY_DSN=your-sentry-dsn
```

### Permissions
The app requires the following permissions:
- **Location**: For accurate crop recommendations
- **Camera**: For soil image capture
- **Storage**: For image selection from gallery

### Theming
The app supports three theme modes:
- **Light**: Bright, clean interface
- **Dark**: Dark mode for low-light conditions
- **System**: Automatically follows device theme
Themes are built with Material Design 3 principles and include agricultural-focused color schemes.

## 📊 API Integration
The app integrates with the following backend endpoints:
- `POST /predict-crop` - Crop prediction with optional image
- `POST /predict-soil` - Soil type detection from image
- `POST /water-scarcity` - Water scarcity analysis
- `GET /crops` - Available crops list
- `GET /health` - Backend health check

## 🔒 Security & Privacy
- Location data is only used for weather analysis
- Images are processed locally and on secure servers
- No personal data is stored permanently
- All API communications use HTTPS in production
- **CORS**: Configured for cross-origin requests
- **Rate Limiting**: Recommended for production
- **Future Authentication**: JWT for user authentication, OAuth for social login

## 🛠️ Development

### Development Tools
- **Code Editor**: VS Code with extensions:
  - React Native Tools
  - Python
  - Expo Tools
  - TypeScript and JavaScript Language Features
- **Version Control**: Git, GitHub for repository hosting
- **Package Managers**:
  - npm for Node.js packages
  - pip for Python packages
  - Expo CLI for Expo-specific commands

### Adding New Features
1. Create components in `frontend/src/components/`
2. Add screens to `frontend/src/screens/`
3. Update navigation in `frontend/App.tsx`
4. Add API calls to `frontend/src/services/api.ts`

### Testing
#### Frontend Testing
- **Jest**: Unit testing framework
- **Detox**: E2E testing for React Native
- **Expo Go**: Manual testing on devices
```bash
cd frontend
npm test
```

#### Backend Testing
- **pytest**: Python testing framework
- **FastAPI TestClient**: API endpoint testing
- **Postman**: Manual API testing
```bash
cd backend
pytest
```

### Building for Production
#### Frontend
```bash
cd frontend
# Build for production
npx eas build --platform all

# Submit to app stores
npx eas submit --platform all
```

#### Backend
```bash
cd backend
# Using Docker
docker build -t crop-api .
docker run -p 8000:8000 crop-api

# Or direct deployment
pip install -r ../requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📱 Deployment Requirements

### Frontend Deployment
- **Expo Application Services (EAS)**: For building and deploying
- **App Store Connect**: For iOS app distribution
- **Google Play Console**: For Android app distribution

### Backend Deployment
- **Cloud Platform**: AWS, Google Cloud, or Azure
- **Container**: Docker (optional)
- **Database**: PostgreSQL or MongoDB (if needed)
- **File Storage**: AWS S3 or similar for model artifacts

## 📊 Performance Requirements
### Frontend
- **Bundle Size**: < 50MB for optimal performance
- **Load Time**: < 3 seconds on 4G networks
- **Memory Usage**: < 200MB on mobile devices
### Backend
- **Response Time**: < 2 seconds for predictions
- **Concurrent Users**: 100+ simultaneous requests
- **Uptime**: 99.9% availability target

## 🌐 Browser Compatibility (Web Version)
- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+
- **Mobile Browsers**:
  - Chrome Mobile: 90+
  - Safari Mobile: 14+
  - Samsung Internet: 14+

## 📱 Minimum Device Requirements
- **Android**:
  - OS Version: Android 5.0 (API level 21)
  - RAM: 2GB minimum
  - Storage: 100MB free space
- **iOS**:
  - OS Version: iOS 13.4
  - Device: iPhone 6s or newer
  - Storage: 100MB free space

## 🔍 Additional Tools & Services
- **Monitoring & Analytics**:
  - Sentry: Error tracking and performance monitoring
  - Google Analytics: User behavior tracking
  - Crashlytics: Crash reporting
- **CI/CD Pipeline**:
  - GitHub Actions: Automated testing and deployment
  - EAS Build: Automated app building
  - Fastlane: iOS/Android deployment automation

## 📄 License Requirements
- **Open Source Libraries**: All dependencies use MIT, Apache 2.0, or BSD licenses
- **Commercial Use**:
  - Expo: Free tier available, paid plans for advanced features
  - Cloud services: Pay-as-you-use pricing models

## 🆘 Troubleshooting

### Common Issues
1. **Expo Version Mismatch**
   ```bash
   cd frontend
   npx expo install --fix
   ```

2. **Python Dependencies Conflict**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r ../requirements.txt
   ```

3. **Port Already in Use**
   ```bash
   npx kill-port 8000
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

4. **Network Connection Issues**
   ```bash
   cd frontend
   npx expo start --tunnel
   # Or
   npx expo start --lan
   ```

5. **Metro Bundler Issues**
   ```bash
   cd frontend
   npx react-native start --reset-cache
   ```

6. **Android Build Issues**
   ```bash
   cd frontend/android
   ./gradlew clean
   cd ..
   npm run android
   ```

7. **iOS Build Issues**
   ```bash
   cd frontend/ios
   rm -rf Pods
   pod install
   cd ..
   npm run ios
   ```

8. **Package Conflicts**
   ```bash
   cd frontend
   rm -rf node_modules
   npm install
   npm cache clean --force
   ```

### Success Indicators
✅ Backend server running on port 8000  
✅ React Native Metro bundler started  
✅ App launches without errors  
✅ Navigation between screens works  
✅ API calls to backend successful  

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Support & Resources
- **Documentation**:
  - [Expo Documentation](https://docs.expo.dev/)
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
  - [React Native Documentation](https://reactnative.dev/)
- **Community**:
  - [Expo Discord](https://discord.gg/expo)
  - [React Native Community](https://reactnative.dev/community/overview)
  - [FastAPI GitHub](https://github.com/tiangolo/fastapi)
- **Issues**:
  - Check this troubleshooting guide
  - Search existing GitHub issues
  - Create a new issue with detailed error logs