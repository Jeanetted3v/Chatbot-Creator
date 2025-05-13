// src/frontend/src/app/components/chat/MessageInput.tsx
'use client';

import { useState } from 'react';
import { Send } from 'lucide-react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  isStaffMode: boolean;
  placeholder?: string;
  isDisabled?: boolean;
}

export default function MessageInput({ 
  onSendMessage, 
  isStaffMode, 
  placeholder,
  isDisabled = false 
}: MessageInputProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isDisabled) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder={placeholder || `Type your message as ${isStaffMode ? 'staff' : 'user'}...`}
        className={`flex-1 p-2 border border-gray-300 rounded-md focus:outline-none focus:border-blue-500 ${
          isDisabled ? 'bg-gray-100 cursor-not-allowed' : ''
        }`}
        disabled={isDisabled}
      />
      <button
        type="submit"
        className={`p-2 rounded-md flex items-center justify-center ${
          isDisabled 
            ? 'bg-gray-300 cursor-not-allowed' 
            : isStaffMode 
              ? 'bg-blue-500 hover:bg-blue-600 text-white' 
              : 'bg-green-500 hover:bg-green-600 text-white'
        }`}
        disabled={isDisabled}
      >
        <Send className="w-5 h-5" />
      </button>
    </form>
  );
}