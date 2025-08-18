import React, { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Send, Loader2, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const MOOD_SUGGESTIONS = [
    "I want something light and healthy",
    "Craving spicy and bold flavors",
    "Need some warm comfort food",
    "Looking for a romantic dinner",
    "Something quick for lunch",
    "I feel like celebrating!"
];

export default function CentralInput({ onSearch, isLoading, hasSearched }) {
    const [query, setQuery] = useState("");
    const [showSuggestions, setShowSuggestions] = useState(false);
    const inputRef = useRef(null);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim() && !isLoading) {
            onSearch(query.trim());
            setQuery("");
            setShowSuggestions(false);
        }
    };

    const handleSuggestionClick = (suggestion) => {
        setQuery(suggestion);
        setShowSuggestions(false);
        onSearch(suggestion);
    };

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (inputRef.current && !inputRef.current.contains(event.target)) {
                setShowSuggestions(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className="relative w-full max-w-3xl mx-auto" ref={inputRef}>
            <form onSubmit={handleSubmit} className="relative">
                <div className="relative">
                    <Input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onFocus={() => !hasSearched && setShowSuggestions(true)}
                        placeholder={hasSearched ? "Refine your search or try something new..." : "What are you craving today?"}
                        className="pl-12 pr-16 py-6 text-lg bg-white/90 backdrop-blur-sm border-2 border-orange-200 focus:border-orange-400 rounded-2xl shadow-lg transition-all duration-300"
                        disabled={isLoading}
                    />
                    <Sparkles className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-orange-400" />
                    <Button
                        type="submit"
                        size="sm"
                        disabled={!query.trim() || isLoading}
                        className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-orange-500 hover:bg-orange-600 rounded-xl px-4 py-2"
                    >
                        {isLoading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Send className="w-4 h-4" />
                        )}
                    </Button>
                </div>
            </form>

            <AnimatePresence>
                {showSuggestions && !isLoading && !hasSearched && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full mt-2 w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl border border-orange-200 p-6 z-50"
                    >
                        <h3 className="text-sm font-medium text-gray-600 mb-4">Try these suggestions:</h3>
                        <div className="flex flex-wrap gap-2">
                            {MOOD_SUGGESTIONS.map((suggestion, index) => (
                                <Badge
                                    key={index}
                                    variant="outline"
                                    className="cursor-pointer hover:bg-orange-50 hover:border-orange-300 transition-colors px-3 py-2 text-sm"
                                    onClick={() => handleSuggestionClick(suggestion)}
                                >
                                    {suggestion}
                                </Badge>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
