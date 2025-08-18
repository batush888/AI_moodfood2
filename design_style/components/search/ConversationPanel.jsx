import React, { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Send, Loader2, Bot, User as UserIcon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const MOOD_SUGGESTIONS = [
    "I want something light and healthy",
    "Craving spicy and bold flavors",
    "Need some warm comfort food",
    "Looking for a romantic dinner spot",
    "Something quick and cheap for lunch",
    "I feel like celebrating!",
];

export default function ConversationPanel({ conversation, onSendMessage, isLoading }) {
    const [inputMessage, setInputMessage] = useState("");
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [conversation]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (inputMessage.trim() && !isLoading) {
            onSendMessage(inputMessage.trim());
            setInputMessage("");
        }
    };

    const handleSuggestionClick = (suggestion) => {
        if (!isLoading) {
            onSendMessage(suggestion);
        }
    };

    return (
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 h-full flex flex-col">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
                <AnimatePresence>
                    {conversation.length === 0 ? (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex flex-col items-center justify-center h-full text-center"
                        >
                            <Bot className="w-10 h-10 text-orange-400 mx-auto mb-3" />
                            <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Food Assistant</h3>
                            <p className="text-gray-500 mb-4 text-sm">Start a conversation or pick a suggestion below.</p>
                            <div className="flex flex-wrap gap-2 justify-center max-w-md">
                                {MOOD_SUGGESTIONS.map((suggestion, index) => (
                                    <Badge
                                        key={index}
                                        variant="outline"
                                        className="cursor-pointer hover:bg-orange-50 hover:border-orange-300 transition-colors px-3 py-1 text-sm"
                                        onClick={() => handleSuggestionClick(suggestion)}
                                    >
                                        {suggestion}
                                    </Badge>
                                ))}
                            </div>
                        </motion.div>
                    ) : (
                        conversation.map((message, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'
                                    }`}
                            >
                                {message.role === 'assistant' && (
                                    <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0">
                                        <Bot className="w-4 h-4 text-orange-600" />
                                    </div>
                                )}

                                <div
                                    className={`max-w-[75%] rounded-lg p-3 ${message.role === 'user'
                                            ? 'bg-orange-500 text-white'
                                            : 'bg-gray-100 text-gray-900'
                                        }`}
                                >
                                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                                </div>

                                {message.role === 'user' && (
                                    <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center flex-shrink-0">
                                        <UserIcon className="w-4 h-4 text-gray-600" />
                                    </div>
                                )}
                            </motion.div>
                        ))
                    )}
                </AnimatePresence>

                {isLoading && conversation.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex gap-3 justify-start"
                    >
                        <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                            <Bot className="w-4 h-4 text-orange-600" />
                        </div>
                        <div className="bg-gray-100 rounded-lg p-3">
                            <div className="flex gap-1">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                        </div>
                    </motion.div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t border-gray-200 p-4">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <Input
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder={conversation.length === 0 ? "What are you craving today?" : "Not quite right? Add more details..."}
                        className="flex-1 bg-gray-50 border-gray-200 focus:border-orange-400"
                        disabled={isLoading}
                    />
                    <Button
                        type="submit"
                        disabled={!inputMessage.trim() || isLoading}
                        className="bg-orange-500 hover:bg-orange-600 px-4"
                    >
                        {isLoading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Send className="w-4 h-4" />
                        )}
                    </Button>
                </form>
            </div>
        </div>
    );
}
