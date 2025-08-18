import React, { useState, useEffect } from "react";
import { InvokeLLM } from "@/integrations/Core";
import { Restaurant } from "@/entities/Restaurant";
import { UserPreference, User } from "@/entities/all";
import { motion, AnimatePresence } from "framer-motion";

import CentralInput from "../components/search/CentralInput";
import FilterSection from "../components/search/FilterSection";
import FoodRecommendationCard from "../components/search/FoodRecommendationCard";

export default function Home() {
    const [foodRecommendations, setFoodRecommendations] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [activeFilter, setActiveFilter] = useState("all");
    const [currentUser, setCurrentUser] = useState(null);
    const [userPreferences, setUserPreferences] = useState(null);

    useEffect(() => {
        const loadUser = async () => {
            try {
                const user = await User.me();
                setCurrentUser(user);

                const preferences = await UserPreference.filter({ user_email: user.email });
                if (preferences.length > 0) {
                    setUserPreferences(preferences[0]);
                }
            } catch (error) {
                // User not logged in
            }
        };
        loadUser();
    }, []);

    const handleSearch = async (query) => {
        setIsLoading(true);
        setHasSearched(true);

        try {
            // Get user preferences context
            const dietaryInfo = userPreferences?.dietary_restrictions?.length > 0
                ? `User's dietary restrictions: ${userPreferences.dietary_restrictions.join(', ')}`
                : "";

            const favoriteInfo = userPreferences?.favorite_cuisines?.length > 0
                ? `User's favorite cuisines: ${userPreferences.favorite_cuisines.join(', ')}`
                : "";

            const contextInfo = [dietaryInfo, favoriteInfo].filter(Boolean).join('\n');

            const analysisResponse = await InvokeLLM({
                prompt: `You are a food recommendation expert. Based on the user's query: "${query}"

${contextInfo ? `User Context:\n${contextInfo}\n\n` : ''}

Provide 3-5 specific food recommendations that match their mood/craving. For each food item, include:
1. Food name and cuisine type
2. Description of the dish
3. Why this food matches their mood/request
4. Key ingredients (3-5 main ingredients)

Focus on specific dishes rather than general categories.`,
                response_json_schema: {
                    type: "object",
                    properties: {
                        recommendations: {
                            type: "array",
                            items: {
                                type: "object",
                                properties: {
                                    name: { type: "string" },
                                    cuisine_type: { type: "string" },
                                    description: { type: "string" },
                                    mood_match_reason: { type: "string" },
                                    ingredients: { type: "array", items: { type: "string" } },
                                    image_keywords: { type: "string" }
                                }
                            }
                        }
                    }
                }
            });

            // Get all restaurants
            const allRestaurants = await Restaurant.list();

            // Match restaurants to each food recommendation
            const enrichedRecommendations = analysisResponse.recommendations.map(food => {
                const matchingRestaurants = allRestaurants
                    .filter(restaurant => {
                        const cuisineMatch = restaurant.cuisine_type?.toLowerCase().includes(food.cuisine_type?.toLowerCase());
                        const dishMatch = restaurant.popular_dishes?.some(dish =>
                            dish.toLowerCase().includes(food.name.toLowerCase()) ||
                            food.name.toLowerCase().includes(dish.toLowerCase())
                        );
                        return cuisineMatch || dishMatch;
                    })
                    .sort((a, b) => b.rating - a.rating)
                    .slice(0, 4);

                return {
                    ...food,
                    restaurants: matchingRestaurants,
                    image_url: getImageUrlForFood(food.name, food.cuisine_type)
                };
            });

            setFoodRecommendations(enrichedRecommendations);

        } catch (error) {
            console.error("Search error:", error);
            setFoodRecommendations([]);
        }

        setIsLoading(false);
    };

    const getImageUrlForFood = (foodName, cuisineType) => {
        const foodImages = {
            "ramen": "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=400&h=300&fit=crop",
            "pasta": "https://images.unsplash.com/photo-1551892374-ecf8dccacd40?w=400&h=300&fit=crop",
            "sushi": "https://images.unsplash.com/photo-1553621042-f6e147245754?w=400&h=300&fit=crop",
            "pizza": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400&h=300&fit=crop",
            "salad": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&h=300&fit=crop",
            "burger": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=300&fit=crop",
            "curry": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=400&h=300&fit=crop",
            "tacos": "https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=400&h=300&fit=crop"
        };

        const foodKey = Object.keys(foodImages).find(key =>
            foodName.toLowerCase().includes(key) || key.includes(foodName.toLowerCase())
        );

        return foodKey ? foodImages[foodKey] : "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop";
    };

    const filteredRecommendations = foodRecommendations.filter(food => {
        if (activeFilter === "all") return true;

        const filterMappings = {
            "quick": ["quick", "fast", "easy"],
            "comfort": ["comfort", "warm", "hearty"],
            "healthy": ["healthy", "light", "fresh"],
            "spicy": ["spicy", "hot", "bold"],
            "sweet": ["sweet", "dessert"],
            "budget": food.restaurants?.some(r => r.price_range === "$"),
            "fine": food.restaurants?.some(r => r.price_range === "$$$$")
        };

        const keywords = filterMappings[activeFilter];
        if (Array.isArray(keywords)) {
            return keywords.some(keyword =>
                food.mood_match_reason?.toLowerCase().includes(keyword) ||
                food.description?.toLowerCase().includes(keyword)
            );
        }
        return keywords;
    });

    return (
        <div className="min-h-screen bg-gradient-to-b from-orange-50 to-white py-8 px-4">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-8"
                >
                    <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
                        What's your
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-orange-600"> craving</span>?
                    </h1>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-8">
                        Tell us your mood and we'll find the perfect dishes for you, along with the best places to get them.
                    </p>
                </motion.div>

                {/* Central Input */}
                <div className="mb-8">
                    <CentralInput
                        onSearch={handleSearch}
                        isLoading={isLoading}
                        hasSearched={hasSearched}
                    />
                </div>

                {/* Filter Section */}
                {hasSearched && (
                    <FilterSection
                        activeFilter={activeFilter}
                        onFilterChange={setActiveFilter}
                    />
                )}

                {/* Loading State */}
                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-center py-12"
                    >
                        <div className="inline-flex items-center gap-3 text-orange-600">
                            <div className="w-8 h-8 border-3 border-orange-600 border-t-transparent rounded-full animate-spin"></div>
                            <span className="text-lg font-medium">Finding your perfect dishes...</span>
                        </div>
                    </motion.div>
                )}

                {/* Food Recommendations */}
                <AnimatePresence>
                    {!isLoading && filteredRecommendations.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="space-y-6"
                        >
                            {filteredRecommendations.map((food, index) => (
                                <FoodRecommendationCard
                                    key={index}
                                    foodItem={food}
                                    index={index}
                                />
                            ))}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Empty State */}
                {!isLoading && hasSearched && filteredRecommendations.length === 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-center py-12"
                    >
                        <p className="text-gray-500 text-lg mb-4">
                            No dishes found matching your criteria.
                        </p>
                        <p className="text-gray-400">
                            Try adjusting your filters or search for something different.
                        </p>
                    </motion.div>
                )}
            </div>
        </div>
    );
}
