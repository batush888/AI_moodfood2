import React from "react";
import { Link, useLocation } from "react-router-dom";
import { createPageUrl } from "@/utils";
import { Search, Heart, User, Settings } from "lucide-react";

export default function Layout({ children, currentPageName }) {
    const location = useLocation();

    return (
        <div className="min-h-screen bg-gradient-to-b from-orange-50 to-white">
            {/* Header */}
            <header className="bg-white/80 backdrop-blur-sm border-b border-orange-100 sticky top-0 z-50">
                <div className="max-w-6xl mx-auto px-4 py-4">
                    <div className="flex items-center justify-between">
                        <Link to={createPageUrl("Home")} className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gradient-to-r from-orange-400 to-orange-500 rounded-full flex items-center justify-center">
                                <span className="text-white font-bold text-lg">🍽️</span>
                            </div>
                            <div>
                                <h1 className="text-xl font-bold text-gray-900">MoodFood</h1>
                                <p className="text-xs text-gray-500">Find food that matches your mood</p>
                            </div>
                        </Link>

                        <nav className="hidden md:flex items-center gap-6">
                            <Link
                                to={createPageUrl("Home")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${location.pathname === createPageUrl("Home")
                                        ? 'bg-orange-100 text-orange-600'
                                        : 'text-gray-600 hover:text-orange-600'
                                    }`}
                            >
                                <Search className="w-4 h-4" />
                                <span className="font-medium">Search</span>
                            </Link>
                            <Link
                                to={createPageUrl("Favorites")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${location.pathname === createPageUrl("Favorites")
                                        ? 'bg-orange-100 text-orange-600'
                                        : 'text-gray-600 hover:text-orange-600'
                                    }`}
                            >
                                <Heart className="w-4 h-4" />
                                <span className="font-medium">Favorites</span>
                            </Link>
                            <Link
                                to={createPageUrl("Profile")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${location.pathname === createPageUrl("Profile")
                                        ? 'bg-orange-100 text-orange-600'
                                        : 'text-gray-600 hover:text-orange-600'
                                    }`}
                            >
                                <User className="w-4 h-4" />
                                <span className="font-medium">Profile</span>
                            </Link>
                        </nav>

                        {/* Mobile menu button */}
                        <div className="md:hidden">
                            <Settings className="w-6 h-6 text-gray-600" />
                        </div>
                    </div>
                </div>
            </header>

            {/* Main content */}
            <main className="flex-1">
                {children}
            </main>

            {/* Mobile bottom navigation */}
            <nav className="md:hidden bg-white border-t border-orange-100 px-4 py-2 fixed bottom-0 left-0 right-0">
                <div className="flex justify-around">
                    <Link
                        to={createPageUrl("Home")}
                        className={`flex flex-col items-center gap-1 py-2 px-4 rounded-lg ${location.pathname === createPageUrl("Home")
                                ? 'text-orange-600'
                                : 'text-gray-500'
                            }`}
                    >
                        <Search className="w-5 h-5" />
                        <span className="text-xs font-medium">Search</span>
                    </Link>
                    <Link
                        to={createPageUrl("Favorites")}
                        className={`flex flex-col items-center gap-1 py-2 px-4 rounded-lg ${location.pathname === createPageUrl("Favorites")
                                ? 'text-orange-600'
                                : 'text-gray-500'
                            }`}
                    >
                        <Heart className="w-5 h-5" />
                        <span className="text-xs font-medium">Favorites</span>
                    </Link>
                    <Link
                        to={createPageUrl("Profile")}
                        className={`flex flex-col items-center gap-1 py-2 px-4 rounded-lg ${location.pathname === createPageUrl("Profile")
                                ? 'text-orange-600'
                                : 'text-gray-500'
                            }`}
                    >
                        <User className="w-5 h-5" />
                        <span className="text-xs font-medium">Profile</span>
                    </Link>
                </div>
            </nav>

            {/* Bottom padding for mobile nav */}
            <div className="h-20 md:h-0"></div>
        </div>
    );
}
