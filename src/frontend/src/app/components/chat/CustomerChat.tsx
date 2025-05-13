'use client';

import { useState, useEffect, useRef } from 'react';
import { ApiService, ChatMessage, getWebSocketUrl } from '../../services/api';
import MessageInput from './MessageInput';

interface CustomerChatProps {
  customerId: string;
  sessionId: string;
}

interface UIMessage {
  id: string;
  content: string;
  sender: 'user' | 'bot' | 'staff' | 'system';
  timestamp: Date;
}

interface ApiMessage {
  role: string;
  content: string;
  timestamp: string;
  customer_id: string;
  session_id: string;
}

// React component for a customer chat interface
export default function CustomerChat({ customerId, sessionId }: CustomerChatProps) {
  // State variables to manage component data
  const [messages, setMessages] = useState<UIMessage[]>([]); // Stores chat messages
  const [loading, setLoading] = useState(true);  // Tracks when data is loading
  const [error, setError] = useState('');  // Stores error messages
  const [socket, setSocket] = useState<WebSocket | null>(null);  // WebSocket connection
  const messagesEndRef = useRef<HTMLDivElement>(null);  // Ref to scroll to bottom
  
  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Sets up the component and defines state variables that will hold all the data needed for the chat to work.
  // Load initial chat history
  useEffect(() => {
    // Add a flag to track if we've received history
    let historyReceived = false;
    setMessages([{
      id: Date.now().toString(),
      content: 'Welcome to our support chat! How can we help you today?',
      sender: 'bot',
      timestamp: new Date()
    }]);
    setLoading(true);
    setError('');
    // Create a timeout fallback in case WebSocket history is delayed or fails
    // const historyTimeout = setTimeout(() => {
    //   // If we're still loading, fall back to HTTP
    //   if (loading && !historyReceived) {
    //     console.log('WebSocket history not received, falling back to HTTP');
    //     fetchHistoryViaHTTP();
    //   }
    // }, 5000); // 3 second timeout

    // Define the HTTP fallback function
    const fetchHistoryViaHTTP = async () => {
      try {
        console.log('Fetching chat history via HTTP fallback');
        const history = await ApiService.getChatHistory(sessionId, customerId);
        
        if (!history || history.length === 0) {
          console.log('No history found via HTTP');
          setLoading(false);
          return; // Keep welcome message if no history
        }
        
        console.log('Received history via HTTP:', history.length, 'messages');
        // Map backend messages to UI format
        const uiMessages = history
          .map((msg: ChatMessage) => ({
            id: `${msg.timestamp}-${Math.random()}`,
            content: msg.content,
            sender: mapRoleToSender(msg.role),
            timestamp: new Date(msg.timestamp)
          }));
        uiMessages.sort((a: UIMessage, b: UIMessage) => a.timestamp.getTime() - b.timestamp.getTime());
        
        if (uiMessages.length > 0) {
          setMessages(uiMessages);
        }
        // If no messages, keep the welcome message we set earlier
        setLoading(false);
      } catch (err) {
        console.error('Error loading chat history via HTTP:', err);
        setError('Could not load chat history');
        setLoading(false);
      }
    };
    // Clean up on unmount
    return () => {
      // clearTimeout(historyTimeout);
      historyReceived = true; // Prevent further history loads
    };
  }, [customerId, sessionId]);


  // Set up WebSocket connection
  useEffect(() => {
    if (!sessionId || !customerId) return;

    const wsUrl = getWebSocketUrl(`/ws/chat/${sessionId}/customer?customer_id=${customerId}`);
    console.log('Attempting to connect to WebSocket:', wsUrl);
    const newSocket = new WebSocket(wsUrl);

    newSocket.onopen = () => {
      console.log('WebSocket connected successfully!');
      // Get history through the 'history' message type
    };
    // When we receive WebSocket messages
    newSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket message received:', data);
      
      if (data.type === 'new_message') {
        // For user messages, only display if they weren't sent by this client
        if (data.message.role === 'user' && data.message.customer_id === customerId) {
          console.log('Ignoring own message echo from server');
          return; // Skip own messages echoed back
        }
        
        // For all other messages (bot/staff/system or messages from other users)
        const newMessage: UIMessage = {
          id: `${data.message.timestamp}-${Math.random()}`,
          content: data.message.content,
          sender: mapRoleToSender(data.message.role),
          timestamp: new Date(data.message.timestamp)
        };
        
        // Enhanced duplicate detection
        setMessages(prev => {
          // Check for duplicates more carefully - consider content similarity
          const isDuplicate = prev.some(msg => {
            // If content is identical and timestamps are close, likely duplicate
            const contentMatch = msg.content === newMessage.content;
            const timeClose = Math.abs(
              new Date(msg.timestamp).getTime() - 
              new Date(newMessage.timestamp).getTime()
            ) < 5000; // 5 seconds window
            
            return contentMatch && timeClose;
          });
          
          if (isDuplicate) {
            console.log('Duplicate message detected, not adding to UI');
            return prev;
          }
          
          console.log('Adding message to UI:', newMessage);
          // Create a new array with all previous messages plus the new one
          const updatedMessages = [...prev, newMessage];
          // Sort all messages by timestamp
          updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
          return updatedMessages;
        });
      } else if (data.type === 'history') {
        setLoading(false);
        // Handle full history update if needed
        const historyMessages = data.messages
          .map((msg: ApiMessage) => ({
            id: `${msg.timestamp}-${Math.random()}`,
            content: msg.content,
            sender: mapRoleToSender(msg.role),
            timestamp: new Date(msg.timestamp)
          }));
        
        // Sort messages by time
        historyMessages.sort((a: UIMessage, b: UIMessage) => a.timestamp.getTime() - b.timestamp.getTime());
        
        // Update our messages state
        if (historyMessages.length > 0) {
          setMessages(historyMessages);
        }
      }
    };
    
    newSocket.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      // Try to reconnect after a delay
      setTimeout(() => {
        if (newSocket.readyState === WebSocket.CLOSED) {
          setSocket(null); // This will trigger useEffect to run again
        }
      }, 5000);
    };
    
    newSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    setSocket(newSocket);
    
    // Clean up on unmount
    return () => {
      if (newSocket.readyState === WebSocket.OPEN) {
        newSocket.close();
      }
    };
  }, [sessionId, customerId]);

  // Helper to map backend roles to UI sender types
  const mapRoleToSender = (role: string): 'user' | 'bot' | 'staff' | 'system' => {
    switch (role?.toUpperCase()) {
      case 'CUSTOMER':
      case 'USER':
        return 'user';
      case 'BOT':
        return 'bot';
      case 'HUMAN_AGENT':
        return 'staff';
      case 'SYSTEM':
        return 'system';
      default:
        return 'system';
    }
  };

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Converts server-side role labels to display-friendly types that the UI can understand.
  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message to UI immediately
    const userMessage: UIMessage = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    // Try WebSocket first
    if (socket && socket.readyState === WebSocket.OPEN) {
      console.log("Sending message via WebSocket");
      socket.send(JSON.stringify({
        type: "message",
        content: content,
        customer_id: customerId,
        session_id: sessionId
      }));
      return; // Exit early - no need to call API
    } 
    
    // Only fall back to API if WebSocket isn't available
    try {
      console.log("WebSocket unavailable, falling back to API");
      const response = await ApiService.sendCustomerMessage(content, customerId, sessionId);
      
      // Add response to UI if WebSocket fails
      const responseMessage: UIMessage = {
        id: `${Date.now()}-response`,
        content: response.message,
        sender: mapRoleToSender(response.role),
        timestamp: new Date(response.timestamp)
      };
        
      setMessages(prev => [...prev, responseMessage]);
      // If WebSocket is active, the response will come through that
    } catch (err) {
      console.error('Error sending message:', err);
      
      // Add error message
      const errorMessage: UIMessage = {
        id: `${Date.now()}-error`,
        content: 'Sorry, there was an error sending your message. Please try again.',
        sender: 'system',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  if (loading) {
    return <div className="flex justify-center items-center h-full">Loading...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-full text-red-500">{error}</div>;
  }

  return (
    <div className="flex flex-col h-[calc(100%-3.5rem)]">
      {/* Debug info */}
      <div className="bg-yellow-100 p-2 text-xs">
        <div>Messages in state: {messages.length}</div>
      </div>
      
      {/* Direct message rendering */}
      <div className="flex-1 overflow-auto p-4">
        {messages.map(message => {
          // Format date and time
          const messageDate = new Date(message.timestamp);
          const formattedDate = messageDate.toLocaleDateString();
          const formattedTime = messageDate.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false // This ensures 24-hour format
          });
          
          return (
            <div 
              key={message.id}
              style={{
                padding: '10px',
                margin: '8px 0',
                borderRadius: '8px',
                backgroundColor: message.sender === 'user' ? '#e6f7ff' : '#f0f0f0',
                border: '2px solid #d9d9d9',
                marginLeft: message.sender === 'user' ? 'auto' : '0',
                marginRight: message.sender === 'user' ? '0' : 'auto',
                maxWidth: '80%'
              }}
            >
              <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '4px' }}>
                <strong>
                  {message.sender === 'user' ? 'Customer' : 
                   message.sender === 'bot' ? 'Bot' : 
                   message.sender === 'staff' ? 'Support Agent' : 'System'}
                </strong> • {formattedDate} {formattedTime}
              </div>
              <div>{message.content}</div>
            </div>
          );
        })}
        
        {/* Comment out MessageList temporarily */}
        {/* <MessageList messages={messages} /> */}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-4 border-t">
        <MessageInput 
          onSendMessage={handleSendMessage} 
          isStaffMode={false} 
          placeholder="Type your message..."
        />
      </div>
    </div>
  );
}