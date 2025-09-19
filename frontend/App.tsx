import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {createStackNavigator} from '@react-navigation/stack';
import {Provider as PaperProvider} from 'react-native-paper';
import { MaterialIcons } from '@expo/vector-icons';
import {StatusBar} from 'react-native';

import {ThemeProvider, useTheme} from './src/context/ThemeContext';
import HomeScreen from './src/screens/HomeScreen';
import CropPredictionScreen from './src/screens/CropPredictionScreen';
import SoilDetectionScreen from './src/screens/SoilDetectionScreen';
import WaterForecastScreen from './src/screens/WaterForecastScreen';
import CropsListScreen from './src/screens/CropsListScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import ResultScreen from './src/screens/ResultScreen';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

function MainTabs() {
  const {theme} = useTheme();
  
  return (
    <Tab.Navigator
      screenOptions={({route}) => ({
        tabBarIcon: ({focused, color, size}) => {
          let iconName = 'home';
          
          if (route.name === 'Home') {
            iconName = 'home';
          } else if (route.name === 'Crop Prediction') {
            iconName = 'agriculture';
          } else if (route.name === 'Soil Detection') {
            iconName = 'terrain';
          } else if (route.name === 'Water Forecast') {
            iconName = 'water';
          } else if (route.name === 'Settings') {
            iconName = 'settings';
          }
          
          return <MaterialIcons name={iconName as any} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.onSurfaceVariant,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.outline,
        },
        headerStyle: {
          backgroundColor: theme.colors.surface,
        },
        headerTintColor: theme.colors.onSurface,
      })}>
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Crop Prediction" component={CropPredictionScreen} />
      <Tab.Screen name="Soil Detection" component={SoilDetectionScreen} />
      <Tab.Screen name="Water Forecast" component={WaterForecastScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

function AppNavigator() {
  const {theme} = useTheme();
  
  return (
    <NavigationContainer theme={theme}>
      <StatusBar
        barStyle={theme.dark ? 'light-content' : 'dark-content'}
        backgroundColor={theme.colors.surface}
      />
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            backgroundColor: theme.colors.surface,
          },
          headerTintColor: theme.colors.onSurface,
        }}>
        <Stack.Screen 
          name="MainTabs" 
          component={MainTabs} 
          options={{headerShown: false}} 
        />
        <Stack.Screen 
          name="Result" 
          component={ResultScreen}
          options={{title: 'Prediction Results'}}
        />
        <Stack.Screen 
          name="CropsList" 
          component={CropsListScreen}
          options={{title: 'Available Crops'}}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <PaperProvider>
        <AppNavigator />
      </PaperProvider>
    </ThemeProvider>
  );
}