// src/frontend/src/app/services/apiService.ts

// Simple UUID generator for demo purposes
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Types matching our backend models
export interface ChatMessage {
  content: string;
  role: 'CUSTOMER' | 'BOT' | 'HUMAN_AGENT' | 'SYSTEM';
  session_id: string;
  customer_id: string;
  timestamp: Date;
}

export interface ChatSession {
  session_id: string;
  customer_id: string;
  current_agent: 'BOT' | 'HUMAN';
  start_time: Date;
  last_interaction: Date;
  message_count: number;
}

export interface MessageRequest {
  message: string;
  customer_id: string;
  session_id?: string;
}

export interface StaffMessageRequest {
  message: string;
  customer_id: string;
  session_id: string;
}

export interface TakeoverRequest {
  session_id: string;
  customer_id: string;
  message?: string;
}

export interface TransferRequest {
  session_id: string;
  customer_id: string;
  message?: string;
}

// Base API URL - adjust as needed
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
export const getWebSocketUrl = (path: string) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // Use the API_URL but replace http:// with ws:// or https:// with wss://
  const wsBase = API_URL.replace(/^http/, 'ws');
  return `${wsBase}${path}`;
};

// API Service class
export class ApiService {
  // Customer endpoints
  static async sendCustomerMessage(message: string, customerId: string, sessionId?: string): Promise<any> {
    try {
      const response = await fetch(`${API_URL}/customer/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          customer_id: customerId,
          session_id: sessionId || '',
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error sending customer message:', error);
      throw error;
    }
  }

  static async getChatHistory(
    sessionId: string,
    customerId: string,
    limit: number = 50
  ): Promise<any> {
    try {
      console.log(`Getting full history for customer: ${customerId}`);
      const response = await fetch(
        `${API_URL}/utils/chat/history?session_id=${sessionId}&customer_id=${customerId}&limit=${limit}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      return data
    } catch (error) {
      console.error('Error getting chat history:', error);
      throw error;
    }
  }

  // Staff endpoints
  static async getActiveSessions(): Promise<ChatSession[]> {
    try {
      const response = await fetch(`${API_URL}/staff/sessions/active`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error getting active sessions:', error);
      throw error;
    }
  }

  static async sendStaffMessage(message: string, sessionId: string, customerId: string): Promise<any> {
    try {
      const response = await fetch(`${API_URL}/staff/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          session_id: sessionId,
          customer_id: customerId,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error sending staff message:', error);
      throw error;
    }
  }

  static async takeOverSession(sessionId: string, customerId: string, message?: string): Promise<any> {
    try {
      const response = await fetch(`${API_URL}/staff/takeover`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          customer_id: customerId,
          message,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error taking over session:', error);
      throw error;
    }
  }

  static async transferToBot(sessionId: string, customerId: string, message?: string): Promise<any> {
    try {
      const response = await fetch(`${API_URL}/staff/transfer/bot`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          customer_id: customerId,
          message,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error transferring to bot:', error);
      throw error;
    }
  }

  // Utility function to generate a random customer ID for demo purposes
  static generateCustomerId(): string {
    return `customer-${generateUUID().substring(0, 8)}`;
  }

  // Utility function to generate a random session ID for demo purposes
  static generateSessionId(): string {
    return `session-${generateUUID()}`;
  }
}