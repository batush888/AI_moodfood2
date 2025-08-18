import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Star, MapPin, Phone, Heart } from "lucide-react";
import { motion } from "framer-motion";

export default function RestaurantCard({ restaurant, index }) {
    const getPriceColor = (price) => {
        switch (price) {
            case "$": return "text-green-600 bg-green-50";
            case "$$": return "text-yellow-600 bg-yellow-50";
            case "$$$": return "text-orange-600 bg-orange-50";
            case "$$$$": return "text-red-600 bg-red-50";
            default: return "text-gray-600 bg-gray-50";
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
        >
            <Card className="overflow-hidden hover:shadow-xl transition-all duration-300 bg-white/80 backdrop-blur-sm border-orange-100 h-full flex flex-col">
                <div className="aspect-video relative overflow-hidden">
                    <img
                        src={restaurant.image_url || `https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=400&h=240&fit=crop&crop=center`}
                        alt={restaurant.name}
                        className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
                    />
                    <div className="absolute top-3 right-3">
                        <Button size="icon" variant="secondary" className="w-8 h-8 bg-white/90 hover:bg-white">
                            <Heart className="w-4 h-4" />
                        </Button>
                    </div>
                    <div className="absolute bottom-3 left-3">
                        <Badge className={`${getPriceColor(restaurant.price_range)} border-0`}>
                            {restaurant.price_range}
                        </Badge>
                    </div>
                </div>

                <div className="flex-1 flex flex-col p-4">
                    <CardHeader className="p-0 pb-3">
                        <div className="flex items-start justify-between">
                            <div>
                                <h3 className="text-lg font-bold text-gray-900 mb-1">{restaurant.name}</h3>
                                <p className="text-sm text-gray-600">{restaurant.cuisine_type}</p>
                            </div>
                            <div className="flex items-center gap-1 bg-orange-50 px-2 py-1 rounded-full flex-shrink-0">
                                <Star className="w-4 h-4 text-orange-500 fill-current" />
                                <span className="text-sm font-medium text-orange-700">{restaurant.rating}</span>
                            </div>
                        </div>
                    </CardHeader>

                    <CardContent className="p-0 flex-1">
                        <p className="text-sm text-gray-600 mb-4 line-clamp-2">{restaurant.description}</p>

                        <div className="space-y-2 text-xs text-gray-500">
                            {restaurant.address && (
                                <div className="flex items-center gap-2">
                                    <MapPin className="w-3 h-3" />
                                    <span className="truncate">{restaurant.address}</span>
                                </div>
                            )}
                            {restaurant.distance_km && (
                                <div className="flex items-center gap-2">
                                    <span className="font-semibold">{restaurant.distance_km} km away</span>
                                </div>
                            )}
                        </div>
                    </CardContent>

                    <div className="mt-4 pt-4 border-t border-gray-100">
                        <Button className="w-full bg-orange-500 hover:bg-orange-600 text-white">
                            View Details
                        </Button>
                    </div>
                </div>
            </Card>
        </motion.div>
    );
}
