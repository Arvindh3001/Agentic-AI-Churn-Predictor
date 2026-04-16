"use client";

import { useEffect, useRef } from "react";
import { WS_URL, getToken } from "@/lib/api";
import type { Customer } from "@/types";

/**
 * Connects to /ws/customers and calls `onUpdate` whenever the backend
 * broadcasts a customer_updated event (e.g. after a PATCH request).
 *
 * Uses a ref pattern so the callback is always fresh without causing
 * the WebSocket to reconnect on every render.
 */
export function useCustomerSocket(onUpdate: (customer: Customer) => void): void {
  const cbRef = useRef(onUpdate);
  // Keep ref current without triggering reconnect
  useEffect(() => {
    cbRef.current = onUpdate;
  });

  useEffect(() => {
    const token = getToken();
    if (!token || typeof window === "undefined") return;

    const ws = new WebSocket(`${WS_URL}/ws/customers`);

    ws.onopen = () => {
      console.debug("[CustomerWS] connected");
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data as string);
        if (msg.type === "customer_updated" && msg.customer) {
          cbRef.current(msg.customer as Customer);
        }
      } catch {
        // Ignore malformed frames
      }
    };

    ws.onerror = () => {
      console.debug("[CustomerWS] connection error — will not retry");
    };

    ws.onclose = () => {
      console.debug("[CustomerWS] closed");
    };

    return () => {
      ws.close();
    };
  }, []); // Mount once; cbRef keeps callback fresh
}
