# GraphQL schema for more flexible querying

type Query {
  # User-related queries
  user(id: ID!): User
  userProfile(userId: ID!): UserProfile
  
  # Food-related queries
  food(id: ID!): FoodItem
  foods(
    query: String
    cuisineType: CuisineType
    dietaryRestrictions: [DietaryRestriction!]
    limit: Int = 20
    offset: Int = 0
  ): [FoodItem!]!
  
  # Recommendation queries
  recommendations(input: RecommendationInput!): RecommendationResponse!
  
  # Restaurant queries
  restaurants(
    location: LocationInput!
    radius: Float = 5.0
    cuisineType: CuisineType
    minRating: Float
  ): [Restaurant!]!
  
  # Analytics queries
  popularFoods(
    timePeriod: String = "7d"
    moodCategory: MoodCategory
    limit: Int = 20
  ): [PopularFood!]!
  
  moodTrends(
    timePeriod: String = "30d"
    granularity: String = "daily"
  ): [MoodTrend!]!
}

type Mutation {
  # User profile mutations
  updateUserProfile(userId: ID!, input: UserProfileInput!): UserProfile!
  
  # Feedback mutations
  submitFeedback(input: FeedbackInput!): FeedbackResponse!
  
  # Admin mutations
  createFoodItem(input: FoodItemInput!): FoodItem!
  updateFoodItem(id: ID!, input: FoodItemInput!): FoodItem!
  createRestaurant(input: RestaurantInput!): Restaurant!
}

type Subscription {
  # Real-time recommendation updates
  recommendationUpdates(userId: ID!): RecommendationResponse!
  
  # Real-time analytics updates
  popularFoodUpdates: [PopularFood!]!
}

# Types
type User {
  id: ID!
  email: String!
  profile: UserProfile
  createdAt: DateTime!
  updatedAt: DateTime!
}

type UserProfile {
  userId: ID!
  ageRange: String
  location: Location
  culturalBackground: [String!]!
  dietaryRestrictions: [DietaryRestriction!]!
  allergies: [String!]!
  healthGoals: [String!]!
  flavorPreferences: FlavorProfile!
  texturePreferences: TextureProfile!
  cuisinePreferences: [CuisinePreference!]!
  moodFoodPatterns: [MoodFoodPattern!]!
  totalInteractions: Int!
  averageRating: Float!
  lastInteraction: DateTime
  createdAt: DateTime!
  updatedAt: DateTime!
}

type FoodItem {
  id: ID!
  name: String!
  cuisineType: CuisineType!
  category: String!
  description: String
  flavorProfile: FlavorProfile!
  textureProfile: TextureProfile!
  nutritionalInfo: NutritionalInfo!
  preparationTime: Int!
  cookingTime: Int!
  difficultyLevel: Int!
  servingSize: String!
  dietaryTags: [DietaryRestriction!]!
  culturalVariants: [CulturalVariant!]!
  seasonalAvailability: [String!]!
  moodAssociations: [MoodAssociation!]!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type RecommendationResponse {
  requestId: ID!
  recommendations: [FoodRecommendation!]!
  intentDetected: ProcessedIntent!
  processingTime: Float!
  totalCandidates: Int!
  filtersApplied: [String!]!
  personalizationApplied: Boolean!
  followUpQuestions: [String!]!
  conversationContext: JSON!
}

type FoodRecommendation {
  foodItem: FoodItem!
  relevanceScore: Float!
  moodMatchScore: Float!
  culturalFitScore: Float!
  dietaryCompatibilityScore: Float!
  personalPreferenceScore: Float!
  contextualAppropriatenessScore: Float!
  explanation: String!
  confidence: Float!
  preparationSuggestions: [String!]!
  pairingSuggestions: [String!]!
  restaurantSuggestions: [RestaurantSuggestion!]!
}

type Restaurant {
  id: ID!
  name: String!
  cuisineTypes: [CuisineType!]!
  location: Location!
  address: String!
  averageRating: Float!
  totalReviews: Int!
  priceRange: Int!
  deliveryAvailable: Boolean!
  pickupAvailable: Boolean!
  estimatedDeliveryTime: Int
  deliveryFee: Float
  minimumOrder: Float
  menuItems: [MenuItemConnection!]!
  operatingHours: JSON!
  isActive: Boolean!
}

# Input types
input RecommendationInput {
  userInput: String!
  userId: ID
  context: UserContextInput!
  preferences: JSON
  maxResults: Int = 20
}

input UserContextInput {
  timeOfDay: String!
  dayOfWeek: String!
  season: String!
  weather: String
  temperature: Float
  location: LocationInput
  socialContext: String!
  occasion: String!
  energyLevel: String!
  hungerLevel: String!
  healthStatus: String!
}

input LocationInput {
  latitude: Float!
  longitude: Float!
  address: String
  city: String
  country: String
}

input FeedbackInput {
  userId: ID!
  recommendationId: ID!
  foodId: ID!
  rating: Int
  thumbsUp: Boolean
  feedbackText: String
  clicked: Boolean = false
  timeSpentViewing: Float = 0.0
  ordered: Boolean = false
  shared: Boolean = false
  dismissed: Boolean = false
  feedbackContext: JSON
}

# Enums
enum CuisineType {
  ITALIAN
  CHINESE
  MEXICAN
  INDIAN
  THAI
  JAPANESE
  FRENCH
  MEDITERRANEAN
  AMERICAN
  KOREAN
}

enum DietaryRestriction {
  VEGETARIAN
  VEGAN
  GLUTEN_FREE
  DAIRY_FREE
  KETO
  PALEO
  HALAL
  KOSHER
}

enum MoodCategory {
  COMFORT_EMOTIONAL
  ENERGY_VITALITY
  WEATHER_SEASONAL
  TEXTURE_SENSORY
  FLAVOR_PROFILE
  OCCASION_SOCIAL
  HEALTH_WELLNESS
}

# Scalar types
scalar DateTime
scalar JSON