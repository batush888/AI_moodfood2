import React, { useState, useEffect } from "react";
import { User } from "@/entities/User";
import { UserPreference } from "@/entities/UserPreference";
import { Heart, Clock, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";

export default function Favorites() {
    const [currentUser, setCurrentUser] = useState(null);
    const [userPreferences, setUserPreferences] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        loadUserData();
    }, []);

    const loadUserData = async () => {
        try {
            const user = await User.me();
            setCurrentUser(user);

            const preferences = await UserPreference.filter({ user_email: user.email });
            if (preferences.length > 0) {
                setUserPreferences(preferences[0]);
            }
        } catch (error) {
            console.error("Error loading user data:", error);
        }
        setIsLoading(false);
    };

    if (isLoading) {
        return (
            <div className="min-h-screen py-8 px-4 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading your preferences...</p>
                </div>
            </div>
        );
    }

    if (!currentUser) {
        return (
            <div className="min-h-screen py-8 px-4 flex items-center justify-center">
                <div className="text-center max-w-md">
                    <Heart className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">Sign in to save favorites</h2>
                    <p className="text-gray-600 mb-6">
                        Create an account to save your favorite restaurants and track your mood-based food preferences.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen py-8 px-4">
            <div className="max-w-4xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">Your Food Journey</h1>
                    <p className="text-gray-600">Track your preferences and discover patterns in your cravings</p>
                </motion.div>

                <div className="grid md:grid-cols-2 gap-6 mb-8">
                    {/* Recent Searches */}
                    <Card className="bg-white/80 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Clock className="w-5 h-5 text-orange-500" />
                                Recent Searches
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            {userPreferences?.search_history?.length > 0 ? (
                                <div className="space-y-3">
                                    {userPreferences.search_history.slice(-5).reverse().map((search, index) => (
                                        <div key={index} className="p-3 bg-gray-50 rounded-lg">
                                            <p className="font-medium text-gray-900 mb-1">"{search.query}"</p>
                                            <p className="text-sm text-gray-500">
                                                {new Date(search.timestamp).toLocaleDateString()}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-gray-500 text-sm">No searches yet. Start exploring!</p>
                            )}
                        </CardContent>
                    </Card>

                    {/* Mood Trends */}
                    <Card className="bg-white/80 backdrop-blur-sm">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-orange-500" />
                                Your Mood Patterns
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            {userPreferences?.preferred_moods?.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {userPreferences.preferred_moods.map((mood, index) => (
                                        <Badge key={index} variant="outline" className="bg-orange-50 border-orange-200">
                                            {mood}
                                        </Badge>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-gray-500 text-sm">Keep searching to see your mood patterns!</p>
                            )}
                        </CardContent>
                    </Card>
                </div>

                {/* Dietary Preferences */}
                <Card className="bg-white/80 backdrop-blur-sm mb-8">
                    <CardHeader>
                        <CardTitle>Dietary Preferences</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {userPreferences?.dietary_restrictions?.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                                {userPreferences.dietary_restrictions.map((restriction, index) => (
                                    <Badge key={index} variant="secondary">
                                        {restriction}
                                    </Badge>
                                ))}
                            </div>
                        ) : (
                            <p className="text-gray-500 text-sm">No dietary preferences set.</p>
                        )}
                    </CardContent>
                </Card>

                {/* Favorite Cuisines */}
                <Card className="bg-white/80 backdrop-blur-sm">
                    <CardHeader>
                        <CardTitle>Favorite Cuisines</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {userPreferences?.favorite_cuisines?.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                                {userPreferences.favorite_cuisines.map((cuisine, index) => (
                                    <Badge key={index} variant="outline" className="bg-green-50 border-green-200">
                                        {cuisine}
                                    </Badge>
                                ))}
                            </div>
                        ) : (
                            <p className="text-gray-500 text-sm">No favorite cuisines discovered yet.</p>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
