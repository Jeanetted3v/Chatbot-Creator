// src/frontend/src/app/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { ApiService, ChatSession } from './services/api';
import CustomerChat from './components/chat/CustomerChat';
import StaffChat from './components/chat/StaffChat';
import { Alert, AlertTitle, AlertDescription } from './components/ui/alert';
import { Info } from 'lucide-react';

export default function ChatDemoPage() {
  const [customerId, setCustomerId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [activeSessions, setActiveSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Initialize the demo 
  useEffect(() => {
    const initDemo = async () => {
      try {
        setLoading(true);
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const sessionResponse = await fetch(`${API_URL}/customer/session/new`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        if (!sessionResponse.ok) {
          throw new Error(`Error: ${sessionResponse.status}`);
        }
        
        const sessionData = await sessionResponse.json();
        setCustomerId(sessionData.customer_id);
        setSessionId(sessionData.session_id);
        
        // Fetch active sessions as before
        const sessions = await ApiService.getActiveSessions();
        setActiveSessions(sessions);
        
        setLoading(false);
      } catch (err) {
        console.error('Error initializing demo:', err);
        setError('Failed to initialize demo. API server may be offline.');
        setLoading(false);
      }
    };
    initDemo();
  }, []);

  // Polling for active sessions (for staff UI)
  useEffect(() => {
    const pollInterval = setInterval(async () => {
      try {
        const sessions = await ApiService.getActiveSessions();
        setActiveSessions(prev => {
          // Only update if there are changes to avoid unnecessary re-renders
          const hasChanged = JSON.stringify(prev) !== JSON.stringify(sessions);
          return hasChanged ? sessions : prev;
        });
      } catch (err) {
        console.error('Error polling sessions:', err);
      }
    }, 120000); // Poll every 120 seconds

    return () => clearInterval(pollInterval);
  }, []);

  if (loading) {
    return <div className="flex justify-center items-center h-screen">Loading...</div>;
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
        <Alert variant="destructive" className="max-w-md">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold text-center mb-6">Chat Support Demo</h1>
      
      <Alert className="mb-4">
        <Info className="h-4 w-4" />
        <AlertTitle>Demo Information</AlertTitle>
        <AlertDescription>
          This demo shows a customer chat UI on the left and a staff support UI on the right.
          <br />
          Customer ID: <code className="bg-gray-100 p-1 rounded">{customerId}</code>
          <br />
          Session ID: <code className="bg-gray-100 p-1 rounded">{sessionId}</code>
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="border rounded-lg p-2 bg-gray-50">
          <h2 className="text-lg font-medium mb-2 text-center">Customer View</h2>
          <div className="bg-white border rounded-lg overflow-hidden" style={{ maxWidth: '400px', margin: '0 auto', height: '600px' }}>
            <div className="h-14 bg-blue-500 text-white flex items-center justify-center">
              <h3 className="font-medium">Fantastic Education Pte. Ltd.</h3>
            </div>
            <CustomerChat 
              customerId={customerId} 
              sessionId={sessionId} 
            />
          </div>
        </div>

        <div className="border rounded-lg p-2 bg-gray-50">
          <h2 className="text-lg font-medium mb-2 text-center">Staff Support View</h2>
          <div className="bg-white border rounded-lg overflow-hidden" style={{ maxWidth: '400px', margin: '0 auto', height: '600px' }}>
            <div className="h-14 bg-gray-700 text-white flex items-center justify-center">
              <h3 className="font-medium">Support Dashboard</h3>
            </div>
            <StaffChat 
              key={`staff-${sessionId}-${customerId}`} 
              selectedSession={{
                session_id: sessionId,
                customer_id: customerId,
                current_agent: 'BOT' as const,
                start_time: new Date(),
                last_interaction: new Date(),
                message_count: 0
              }} 
              activeSessions={activeSessions}
            />
          </div>
        </div>
      </div>
    </div>
  );
}