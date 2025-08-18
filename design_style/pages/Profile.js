import React, { useState, useEffect } from "react";
import { User } from "@/entities/User";
import { UserPreference } from "@/entities/UserPreference";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { User as UserIcon, Save, Plus, X } from "lucide-react";
import { motion } from "framer-motion";

const DIETARY_OPTIONS = [
    "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Paleo",
    "Low-Carb", "Low-Sodium", "Nut-Free", "Halal", "Kosher"
];

export default function Profile() {
    const [currentUser, setCurrentUser] = useState(null);
    const [userPreferences, setUserPreferences] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [selectedDietary, setSelectedDietary] = useState([]);
    const [newCuisine, setNewCuisine] = useState("");

    useEffect(() => {
        loadUserData();
    }, []);

    const loadUserData = async () => {
        try {
            const user = await User.me();
            setCurrentUser(user);

            const preferences = await UserPreference.filter({ user_email: user.email });
            if (preferences.length > 0) {
                const pref = preferences[0];
                setUserPreferences(pref);
                setSelectedDietary(pref.dietary_restrictions || []);
            }
        } catch (error) {
            console.error("Error loading user data:", error);
        }
        setIsLoading(false);
    };

    const handleSavePreferences = async () => {
        if (!currentUser) return;

        setIsSaving(true);
        try {
            const preferenceData = {
                user_email: currentUser.email,
                dietary_restrictions: selectedDietary,
                favorite_cuisines: userPreferences?.favorite_cuisines || [],
                preferred_moods: userPreferences?.preferred_moods || [],
                search_history: userPreferences?.search_history || []
            };

            if (userPreferences) {
                await UserPreference.update(userPreferences.id, preferenceData);
            } else {
                await UserPreference.create(preferenceData);
            }

            await loadUserData();
        } catch (error) {
            console.error("Error saving preferences:", error);
        }
        setIsSaving(false);
    };

    const toggleDietary = (option) => {
        setSelectedDietary(prev =>
            prev.includes(option)
                ? prev.filter(item => item !== option)
                : [...prev, option]
        );
    };

    const addCuisine = () => {
        if (!newCuisine.trim() || !userPreferences) return;

        const updatedPreferences = {
            ...userPreferences,
            favorite_cuisines: [...(userPreferences.favorite_cuisines || []), newCuisine.trim()]
        };
        setUserPreferences(updatedPreferences);
        setNewCuisine("");
    };

    const removeCuisine = (cuisine) => {
        if (!userPreferences) return;

        const updatedPreferences = {
            ...userPreferences,
            favorite_cuisines: userPreferences.favorite_cuisines?.filter(c => c !== cuisine) || []
        };
        setUserPreferences(updatedPreferences);
    };

    if (isLoading) {
        return (
            <div className="min-h-screen py-8 px-4 flex items-center justify-center">
                <div className="text-center">
                    <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading your profile...</p>
                </div>
            </div>
        );
    }

    if (!currentUser) {
        return (
            <div className="min-h-screen py-8 px-4 flex items-center justify-center">
                <div className="text-center max-w-md">
                    <UserIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">Sign in to view profile</h2>
                    <p className="text-gray-600 mb-6">
                        Create an account to customize your food recommendations and save preferences.
                    </p>
                    <Button onClick={() => User.login()} className="bg-orange-500 hover:bg-orange-600">
                        Sign In
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen py-8 px-4">
            <div className="max-w-2xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">Your Profile</h1>
                    <p className="text-gray-600">Customize your food discovery experience</p>
                </motion.div>

                {/* User Info */}
                <Card className="bg-white/80 backdrop-blur-sm mb-6">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <UserIcon className="w-5 h-5 text-orange-500" />
                            Account Information
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <div>
                                <Label>Full Name</Label>
                                <Input value={currentUser.full_name || ""} disabled className="bg-gray-50" />
                            </div>
                            <div>
                                <Label>Email</Label>
                                <Input value={currentUser.email || ""} disabled className="bg-gray-50" />
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Dietary Restrictions */}
                <Card className="bg-white/80 backdrop-blur-sm mb-6">
                    <CardHeader>
                        <CardTitle>Dietary Preferences</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <p className="text-sm text-gray-600">
                                Select your dietary restrictions to get better recommendations
                            </p>
                            <div className="flex flex-wrap gap-2">
                                {DIETARY_OPTIONS.map((option) => (
                                    <Badge
                                        key={option}
                                        variant={selectedDietary.includes(option) ? "default" : "outline"}
                                        className={`cursor-pointer transition-colors ${selectedDietary.includes(option)
                                                ? "bg-orange-500 hover:bg-orange-600"
                                                : "hover:bg-orange-50 hover:border-orange-300"
                                            }`}
                                        onClick={() => toggleDietary(option)}
                                    >
                                        {option}
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Favorite Cuisines */}
                <Card className="bg-white/80 backdrop-blur-sm mb-6">
                    <CardHeader>
                        <CardTitle>Favorite Cuisines</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            <div className="flex gap-2">
                                <Input
                                    placeholder="Add a cuisine you love..."
                                    value={newCuisine}
                                    onChange={(e) => setNewCuisine(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && addCuisine()}
                                />
                                <Button onClick={addCuisine} size="icon" variant="outline">
                                    <Plus className="w-4 h-4" />
                                </Button>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {userPreferences?.favorite_cuisines?.map((cuisine, index) => (
                                    <Badge key={index} variant="secondary" className="flex items-center gap-1">
                                        {cuisine}
                                        <button onClick={() => removeCuisine(cuisine)} className="ml-1 hover:text-red-500">
                                            <X className="w-3 h-3" />
                                        </button>
                                    </Badge>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Save Button */}
                <div className="flex justify-end">
                    <Button
                        onClick={handleSavePreferences}
                        disabled={isSaving}
                        className="bg-orange-500 hover:bg-orange-600"
                    >
                        {isSaving ? (
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                        ) : (
                            <Save className="w-4 h-4 mr-2" />
                        )}
                        Save Preferences
                    </Button>
                </div>
            </div>
        </div>
    );
}
