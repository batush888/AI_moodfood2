import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Star, MapPin } from "lucide-react";
import { motion } from "framer-motion";

export default function FoodRecommendationCard({ foodItem, index }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="mb-8"
        >
            <Card className="overflow-hidden shadow-lg border-orange-100">
                <CardContent className="p-0">
                    {/* Food Item Section */}
                    <div className="flex flex-col md:flex-row">
                        {/* Food Image */}
                        <div className="md:w-1/3">
                            <img
                                src={foodItem.image_url || `https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop`}
                                alt={foodItem.name}
                                className="w-full h-64 md:h-full object-cover"
                            />
                        </div>

                        {/* Food Description */}
                        <div className="md:w-2/3 p-6">
                            <div className="flex items-start justify-between mb-4">
                                <h3 className="text-2xl font-bold text-gray-900">{foodItem.name}</h3>
                                <Badge className="bg-orange-100 text-orange-700 border-orange-200">
                                    {foodItem.cuisine_type}
                                </Badge>
                            </div>

                            <p className="text-gray-700 mb-4 leading-relaxed">{foodItem.description}</p>

                            <div className="mb-4">
                                <h4 className="font-semibold text-gray-900 mb-2">Why this matches your mood:</h4>
                                <p className="text-orange-600 bg-orange-50 p-3 rounded-lg text-sm">
                                    {foodItem.mood_match_reason}
                                </p>
                            </div>

                            <div>
                                <h4 className="font-semibold text-gray-900 mb-2">Key Ingredients:</h4>
                                <div className="flex flex-wrap gap-2">
                                    {foodItem.ingredients?.map((ingredient, idx) => (
                                        <Badge key={idx} variant="outline" className="text-xs">
                                            {ingredient}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Restaurants Section */}
                    <div className="border-t border-gray-200 p-6">
                        <h4 className="text-lg font-bold text-gray-900 mb-4">
                            Restaurants serving {foodItem.name}
                        </h4>
                        <div className="space-y-3">
                            {foodItem.restaurants?.map((restaurant, idx) => (
                                <div
                                    key={idx}
                                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                                >
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-1">
                                            <h5 className="font-semibold text-gray-900">{restaurant.name}</h5>
                                            <div className="flex items-center gap-1">
                                                <Star className="w-4 h-4 text-yellow-500 fill-current" />
                                                <span className="text-sm font-medium">{restaurant.rating}</span>
                                            </div>
                                            <Badge className="text-xs bg-green-100 text-green-700">
                                                {restaurant.price_range}
                                            </Badge>
                                        </div>
                                        <div className="flex items-center gap-4 text-sm text-gray-600">
                                            <div className="flex items-center gap-1">
                                                <MapPin className="w-3 h-3" />
                                                <span>{restaurant.distance_km} km away</span>
                                            </div>
                                            <span>{restaurant.address}</span>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <button className="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                                            Order Now
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    );
}
