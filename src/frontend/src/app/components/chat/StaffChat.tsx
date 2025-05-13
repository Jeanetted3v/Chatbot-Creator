'use client';

import { useState, useEffect, useRef } from 'react';
import { ApiService, ChatMessage, ChatSession, getWebSocketUrl } from '../../services/api';
import MessageInput from './MessageInput';
import { Button } from '../ui/button';
import { User, Bot, RefreshCw } from 'lucide-react';

interface StaffChatProps {
  selectedSession: ChatSession;
  activeSessions: ChatSession[];
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

export default function StaffChat({ selectedSession, activeSessions }: StaffChatProps) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isHumanMode, setIsHumanMode] = useState(selectedSession.current_agent === 'HUMAN');
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [socketStatus, setSocketStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [recentlySentMessages, setRecentlySentMessages] = useState<Set<string>>(new Set());
  
  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Load chat history
  const loadChatHistory = async () => {
    try {
      setLoading(true);
      console.log('Loading chat history for customer:', selectedSession.customer_id);
      
      const history = await ApiService.getChatHistory(
        selectedSession.session_id, 
        selectedSession.customer_id
      );
      
      console.log('Received chat history:', history.length, 'messages');
      
      const uiMessages = history
        .filter((msg: ChatMessage) => 
          msg.customer_id === selectedSession.customer_id
        )
        .map((msg: ChatMessage) => ({
          id: `${msg.timestamp}-${Math.random()}`,
          content: msg.content,
          sender: mapRoleToSender(msg.role),
          timestamp: new Date(msg.timestamp)
        }));

      uiMessages.sort((a: UIMessage, b: UIMessage) => 
        a.timestamp.getTime() - b.timestamp.getTime()
      );
      
      const uniqueMessages = new Map();
      
      for (const msg of uiMessages) {
        const timeWindow = Math.floor(msg.timestamp.getTime() / 10000) * 10000;
        const signature = `${msg.content}|${msg.sender}|${timeWindow}`;
        
        uniqueMessages.set(signature, msg);
      }
      
      setMessages(Array.from(uniqueMessages.values()));
      setLoading(false);
    } catch (err) {
      console.error('Error loading chat history:', err);
      setError('Could not load chat history');
      setLoading(false);
    }
  };

  // Load initial chat history
  useEffect(() => {
    setMessages([]);
    setIsHumanMode(selectedSession.current_agent === 'HUMAN');
    
    if (socket && socket.readyState === WebSocket.OPEN) {
      console.log('Closing existing WebSocket connection');
      socket.close();
    }
    
    setSocket(null);
    setSocketStatus('disconnected');
    setLoading(true); // Set loading until we get history
  }, [selectedSession.session_id, selectedSession.customer_id]);

  // Set up WebSocket connection
  useEffect(() => {
    console.log('StaffChat: WebSocket setup effect running, loading:', loading, 'session ID:', selectedSession.session_id);
    if (!selectedSession.session_id) {
      console.log('Not setting up WebSocket - still loading or no session ID');
      return;
    }
    // Set initial loading state
    setLoading(true);
    
    // Add initial placeholder while waiting for WebSocket history
    setMessages([{
      id: Date.now().toString(),
      content: 'Loading conversation history...',
      sender: 'system',
      timestamp: new Date()
    }]);

    const wsUrl = getWebSocketUrl(`/ws/chat/${selectedSession.session_id}/staff?customer_id=${selectedSession.customer_id}`);
    console.log('Attempting to connect WebSocket to:', wsUrl);
    setSocketStatus('connecting');
    
    const newSocket = new WebSocket(wsUrl);
    
    newSocket.onopen = () => {
      console.log('WebSocket connected successfully for staff view!');
      setSocketStatus('connected');
      // History will be sent automatically by the server
    };
    
    newSocket.onmessage = (event) => {
      console.log('Staff WebSocket message received');
      
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'new_message') {
          if (data.message.customer_id !== selectedSession.customer_id) {
            console.warn('Message filtered out - customer mismatch');
            return;
          }

          const msgRole = data.message.role?.toUpperCase() || '';
          const msgContent = data.message.content || '';
          
          // Prevent duplicate staff messages
          if (msgRole === 'HUMAN_AGENT' && recentlySentMessages.has(msgContent.trim())) {
            console.log('Skipping own staff message');
            return;
          }

          const newMessage: UIMessage = {
            id: `${data.message.timestamp}-${data.message.role}-${data.message.content.substring(0, 20)}`,
            content: data.message.content,
            sender: mapRoleToSender(data.message.role),
            timestamp: new Date(data.message.timestamp)
          };
          
          setMessages(prev => {
            // Check for duplicates
            const isDuplicate = prev.some(msg => {
              const contentMatch = msg.content === newMessage.content;
              const senderMatch = msg.sender === newMessage.sender;
              const timeClose = Math.abs(
                new Date(msg.timestamp).getTime() - 
                new Date(newMessage.timestamp).getTime()
              ) < 30000; // 30 seconds window
              
              return contentMatch && senderMatch && timeClose;
            });
            
            if (isDuplicate) {
              console.log('Duplicate message detected, not adding to UI');
              return prev;
            }
            
            const updatedMessages = [...prev, newMessage];
            updatedMessages.sort((a: UIMessage, b: UIMessage) => a.timestamp.getTime() - b.timestamp.getTime());
            return updatedMessages;
          });
        } else if (data.type === 'history') {
          // We received history from WebSocket - stop loading
          setLoading(false);
          // Filter history messages for current session
          const filteredHistory = data.messages.filter(
            (msg: ApiMessage) =>
              msg.customer_id === selectedSession.customer_id
          );
          console.log('Processing history update from WebSocket:', filteredHistory.length, 'messages');
          
          if (filteredHistory.length > 0) {
            // Convert to UI message format
            const historyMessages = filteredHistory.map((msg: ApiMessage) => ({
              id: `${msg.timestamp}-${Math.random()}`,
              content: msg.content,
              sender: mapRoleToSender(msg.role),
              timestamp: new Date(msg.timestamp)
            }));
            
            // Sort by timestamp
            historyMessages.sort((a: UIMessage, b: UIMessage) => a.timestamp.getTime() - b.timestamp.getTime());
            
            // Set the messages
            setMessages(historyMessages);
          } 
        }  else if (data.type === 'agent_change') {
          console.log('Agent changed to:', data.current_agent);
          setIsHumanMode(data.current_agent === 'HUMAN');
        } else {
          console.log('Unknown message type:', data.type);
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err, event.data);
      }
    };
    
    newSocket.onclose = (event) => {
      console.log('Staff WebSocket disconnected:', event.code, event.reason);
      setSocketStatus('disconnected');
      // Try to reconnect if it wasn't a normal close
      if (event.code !== 1000) {
        setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...');
          setSocket(null);
        }, 5000);
      }
    };
    
    newSocket.onerror = () => {
      console.warn('Staff WebSocket connection failed:');
      setSocketStatus('disconnected');
    };
    
    setSocket(newSocket);
    // Fall back to HTTP if WebSocket takes too long
    const historyTimeout = setTimeout(() => {
      if (loading) {
        console.log('WebSocket history taking too long, falling back to HTTP');
        loadChatHistory(); // Your existing loadChatHistory function
      }
    }, 5000);
    return () => {
      clearTimeout(historyTimeout);
      if (newSocket.readyState === WebSocket.OPEN || newSocket.readyState === WebSocket.CONNECTING) {
        newSocket.close();
      }
    };
  }, [selectedSession.session_id, selectedSession.customer_id]);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Helper to map backend roles to UI sender types
  const mapRoleToSender = (role: string): 'user' | 'bot' | 'staff' | 'system' => {
    role = role?.toUpperCase() || '';
    
    switch (role) {
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
        console.log('Unknown role type:', role);
        return 'system';
    }
  };

  // Handle staff sending a message
  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    try {
      const wasHumanMode = isHumanMode;
      
      if (!isHumanMode) {
        console.log('Not in human mode, taking over...');
        await handleTakeOver();
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        if (!isHumanMode) {
          console.error('Failed to enter human mode');
          return;
        }
      }
      
      console.log('Sending staff message:', content);
      setRecentlySentMessages(prev => {
        const newSet = new Set(prev);
        newSet.add(content.trim());
        
        setTimeout(() => {
          setRecentlySentMessages(current => {
            const updatedSet = new Set(current);
            updatedSet.delete(content.trim());
            return updatedSet;
          });
        }, 30000);
        
        return newSet;
      });
      
      if (wasHumanMode) {
        const staffMessage: UIMessage = {
          id: Date.now().toString(),
          content,
          sender: 'staff',
          timestamp: new Date(),
        };

        setMessages(prev => {
          const updatedMessages = [...prev, staffMessage];
          updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
          return updatedMessages;
        });
      }
      
      if (socket && socket.readyState === WebSocket.OPEN) {
        console.log('Sending via WebSocket');
        
        socket.send(JSON.stringify({
          type: "message",
          content: content,
          customer_id: selectedSession.customer_id,
          session_id: selectedSession.session_id,
          client_message_id: `client-${Date.now()}`
        }));
        
        console.log('Message sent via WebSocket');
        return;
      }
      
      console.log('WebSocket unavailable, falling back to API');
      
      await ApiService.sendStaffMessage(
        content, 
        selectedSession.session_id, 
        selectedSession.customer_id
      );
      
      console.log('Message sent successfully via API');
    } catch (err) {
      console.error('Error sending staff message:', err);
      
      const errorMessage: UIMessage = {
        id: `${Date.now()}-error`,
        content: 'Failed to send message. Please try again.',
        sender: 'system',
        timestamp: new Date()
      };
      
      setMessages(prev => {
        const updatedMessages = [...prev, errorMessage];
        updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
        return updatedMessages;
      });
    }
  };

  // Handle taking over the conversation
  const handleTakeOver = async () => {
    try {
      console.log('Taking over conversation');
      
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
          type: "command",
          action: "takeover",
          session_id: selectedSession.session_id,
          customer_id: selectedSession.customer_id
        }));
        
        console.log('Takeover command sent via WebSocket');
      } else {
        console.log('WebSocket not available, using API fallback');
        await ApiService.takeOverSession(
          selectedSession.session_id, 
          selectedSession.customer_id
        );
        
        const takeoverMessage: UIMessage = {
          id: `${Date.now()}-takeover`,
          content: '--- Human Agent mode activated ---',
          sender: 'system',
          timestamp: new Date()
        };
        
        setMessages(prev => {
          const updatedMessages = [...prev, takeoverMessage];
          updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
          return updatedMessages;
        });
      }
      
      setIsHumanMode(true);
    } catch (err) {
      console.error('Error taking over session:', err);
      
      const errorMessage: UIMessage = {
        id: `${Date.now()}-error`,
        content: 'Failed to take over conversation. Please try again.',
        sender: 'system',
        timestamp: new Date()
      };
      setMessages(prev => {
        const updatedMessages = [...prev, errorMessage];
        updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
        return updatedMessages;
      });
    }
  };

  // Handle transferring back to bot
  const handleTransferToBot = async () => {
    try {
      console.log('Transferring to bot');
      
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
          type: "command",
          action: "transfer_to_bot",
          session_id: selectedSession.session_id,
          customer_id: selectedSession.customer_id
        }));
        
        console.log('Transfer command sent via WebSocket');
      } else {
        console.log('WebSocket not available, using API fallback');
        await ApiService.transferToBot(
          selectedSession.session_id, 
          selectedSession.customer_id
        );
        
        const transferMessage: UIMessage = {
          id: `${Date.now()}-transfer`,
          content: '--- Bot mode activated ---',
          sender: 'system',
          timestamp: new Date()
        };
        
        setMessages(prev => {
          const updatedMessages = [...prev, transferMessage];
          updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
          return updatedMessages;
        });
      }
      
      setIsHumanMode(false);
    } catch (err) {
      console.error('Error transferring to bot:', err);
      
      const errorMessage: UIMessage = {
        id: `${Date.now()}-error`,
        content: 'Failed to transfer to bot. Please try again.',
        sender: 'system',
        timestamp: new Date()
      };
      
      setMessages(prev => {
        const updatedMessages = [...prev, errorMessage];
        updatedMessages.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
        return updatedMessages;
      });
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
      <div className="p-2 border-b bg-gray-50 flex justify-between items-center">
        <div>
          <div className="font-medium">Customer: {selectedSession.customer_id}</div>
          <div className="text-xs text-gray-500">
            Session: {selectedSession.session_id.substring(0, 8)}... | 
            Messages: {selectedSession.message_count} | 
            WebSocket: {
              socketStatus === 'connected' ? 'Connected' : 
              socketStatus === 'connecting' ? 'Connecting...' : 
              'Disconnected'
            }
          </div>
        </div>
        <div className="flex gap-2">
          {isHumanMode ? (
            <Button 
              onClick={handleTransferToBot}
              size="sm"
              variant="outline"
              className="flex items-center gap-1"
            >
              <Bot className="w-4 h-4" />
              <span>Transfer to Bot</span>
            </Button>
          ) : (
            <Button 
              onClick={handleTakeOver}
              size="sm"
              variant="default"
              className="flex items-center gap-1 bg-blue-500 hover:bg-blue-600"
            >
              <User className="w-4 h-4" />
              <span>Take Over</span>
            </Button>
          )}
          <Button
            onClick={loadChatHistory}
            size="sm"
            variant="ghost"
            className="w-8 h-8 p-0"
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>
      
      <div className="flex-1 overflow-auto p-4">
        {messages.map(message => {
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
        
        <div ref={messagesEndRef} />
      </div>
      <div className="p-4 border-t">
        <MessageInput 
          onSendMessage={handleSendMessage} 
          isStaffMode={true} 
          placeholder="Type as Support agent..."
          isDisabled={!isHumanMode}
        />
        {!isHumanMode && (
          <div className="text-xs text-center mt-2 text-gray-500">
            Click 'Take Over' to respond as a human agent
          </div>
        )}
      </div>
    </div>
  );
}