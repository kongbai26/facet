import { createParser, type EventSourceMessage } from 'eventsource-parser'

export async function readSseStream(
  response: Response,
  onEvent: (event: EventSourceMessage) => void,
) {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || response.statusText)
  }
  if (!response.body) {
    throw new Error('Streaming response has no body')
  }

  const decoder = new TextDecoder()
  let receivedTerminalEvent = false
  const parser = createParser({
    onEvent(event) {
      // Our chat SSE contract ends with an explicit ``done`` event. Treating
      // a raw socket EOF as success left a partial/stale assistant message in
      // the UI after proxies or local servers dropped a stream.
      if (event.event === 'done' || event.data === '[DONE]') receivedTerminalEvent = true
      onEvent(event)
    },
  })
  const reader = response.body.getReader()

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      parser.feed(decoder.decode(value, { stream: true }))
    }
    parser.reset({ consume: true })
    if (!receivedTerminalEvent) {
      throw new Error('流式连接意外中断，请重试。')
    }
  } finally {
    reader.releaseLock()
  }
}

export function parseJsonEvent<T>(event: EventSourceMessage): T | null {
  if (!event.data || event.data === '[DONE]') return null
  return JSON.parse(event.data) as T
}
