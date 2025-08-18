import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import RestaurantCard from "./RestaurantCard";
import { ChefHat, Search } from "lucide-react";

export default function ResultsPanel({ restaurants, isLoading, hasSearched }) {
    if (!hasSearched) {
        return (
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 h-full flex items-center justify-center">
                <div className="text-center max-w-md">
                    <ChefHat className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">Ready to Find Your Perfect Meal?</h3>
                    <p className="text-gray-600">
                        Start a conversation above and I'll show you personalized restaurant recommendations based on your mood and preferences.
                    </p>
                </div>
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 h-full flex items-center justify-center">
                <div className="text-center">
                    <div className="w-12 h-12 border-3 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-lg font-medium text-orange-600">Finding perfect matches...</p>
                </div>
            </div>
        );
    }

    if (!restaurants || restaurants.length === 0) {
        return (
            <div className="bg-white rounded-xl shadow-lg border border-gray-200 h-full flex items-center justify-center">
                <div className="text-center max-w-md">
                    <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">No matches found</h3>
                    <p className="text-gray-600">
                        Try refining your request in the chat above. Be more specific about your preferences, budget, or dietary restrictions.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 h-full">
            <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-gray-900">Recommended for You</h2>
                    <span className="text-sm text-gray-500">
                        {restaurants.length} restaurant{restaurants.length !== 1 ? 's' : ''} found
                    </span>
                </div>
            </div>

            <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(100% - 80px)' }}>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <AnimatePresence>
                        {restaurants.map((restaurant, index) => (
                            <RestaurantCard
                                key={restaurant.id}
                                restaurant={restaurant}
                                index={index}
                            />
                        ))}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
}
