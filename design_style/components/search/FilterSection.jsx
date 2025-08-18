import React from "react";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";

const FILTER_OPTIONS = [
    { label: "All", value: "all", color: "bg-gray-100 text-gray-700" },
    { label: "Quick & Easy", value: "quick", color: "bg-blue-100 text-blue-700" },
    { label: "Comfort Food", value: "comfort", color: "bg-orange-100 text-orange-700" },
    { label: "Healthy", value: "healthy", color: "bg-green-100 text-green-700" },
    { label: "Spicy", value: "spicy", color: "bg-red-100 text-red-700" },
    { label: "Sweet", value: "sweet", color: "bg-pink-100 text-pink-700" },
    { label: "Budget-Friendly", value: "budget", color: "bg-yellow-100 text-yellow-700" },
    { label: "Fine Dining", value: "fine", color: "bg-purple-100 text-purple-700" }
];

export default function FilterSection({ activeFilter, onFilterChange }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex flex-wrap justify-center gap-2 py-6"
        >
            {FILTER_OPTIONS.map((filter) => (
                <Badge
                    key={filter.value}
                    variant={activeFilter === filter.value ? "default" : "outline"}
                    className={`cursor-pointer px-4 py-2 text-sm font-medium transition-all hover:scale-105 ${activeFilter === filter.value
                            ? "bg-orange-500 text-white border-orange-500"
                            : `${filter.color} border hover:bg-opacity-80`
                        }`}
                    onClick={() => onFilterChange(filter.value)}
                >
                    {filter.label}
                </Badge>
            ))}
        </motion.div>
    );
}
