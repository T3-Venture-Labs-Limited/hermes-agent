---
name: agentmail
description: Give the agent its own email inbox via AgentMail. Show email threads as interactive UI components using email_thread_list and email_reply_compose compositions.
version: 1.0.0
metadata:
  hermes:
    tags: [email, communication, agentmail, generative-ui]
    category: email
---

# AgentMail — Generative UI Integration

## Requirements

- **AgentMail API key** — sign up at https://console.agentmail.to (free tier: 3 inboxes, 3,000 emails/month)
- The API key must be stored in the agent's environment as `AGENTMAIL_API_KEY`

## Setup

1. Use the `env_vars_form` composition to collect the user's AgentMail API key:
   ```
   render_ui(composition="env_vars_form", data={service: "AgentMail", description: "Your AgentMail API key from console.agentmail.to", fields: [{name: "AGENTMAIL_API_KEY", label: "API Key", type: "password"}]})
   ```
2. The `env_vars_submit` action will write the key to the agent's environment automatically.

## Showing Email Threads

### 1. List inbox threads

After calling `list_threads` from the AgentMail tools:

```
render_ui(
  composition="email_thread_list",
  data={
    inbox_id: "<inbox_id>",
    threads: [
      {
        thread_id: "<thread_id>",
        subject: "Re: Project update",
        snippet: "Thanks for the update! I had a question about...",
        from: "Alice Smith",
        date: "2026-04-03T10:30:00Z",
        read: false,
        labels: ["inbox"]
      }
    ],
    actions: [
      { action: "Compose", label: "Compose New Email" },
      { action: "Refresh", label: "Refresh" }
    ]
  }
)
```

### 2. User opens a thread

When the user clicks on a thread item in `email_thread_list`, the action dispatched is `open_thread`. Since the agent needs to fetch the full thread content and generate a draft reply:

1. Call `get_thread(inbox_id="<inbox_id>", thread_id="<thread_id>")` to get the full thread
2. Generate a draft reply based on the thread context
3. Re-render with `email_reply_compose`:

```
render_ui(
  composition="email_reply_compose",
  data={
    thread_id: "<thread_id>",
    message_id: "<latest_message_id>",
    inbox_id: "<inbox_id>",
    subject: "Re: Project update",
    to: "alice@example.com",
    draft_reply: "Thanks for your email!\n\n...",
    actions: [
      { action: "send_reply", label: "Send Reply" },
      { action: "Discard", label: "Discard" }
    ]
  }
)
```

## Email Actions

These actions are handled directly by the backend — do NOT implement them with tools:

| User action | Backend handler | What happens |
|-------------|----------------|--------------|
| Click "mark_read" on a thread | `_handle_email_mark_read` | Calls AgentMail API to add "read" label |
| Click "mark_unread" on a thread | `_handle_email_mark_unread` | Calls AgentMail API to remove "read" label |
| Click "archive" on a thread | `_handle_email_archive` | Calls AgentMail API to add "archived" label |
| Click "Send Reply" in compose form | `_handle_email_send_reply` | Calls AgentMail API to send the reply |
| Click "Discard" | `_handle_email_discard` | Confirms discard, no API call |

## Reply Flow

1. User clicks a thread → `open_thread` action → agent fetches thread, generates draft
2. Agent renders `email_reply_compose` with draft pre-filled
3. User edits the body and clicks "Send Reply"
4. Backend handler calls `POST /inboxes/{inbox_id}/messages/{message_id}/reply` directly
5. Success → component shows confirmation

## API Key

- Free tier: 3 inboxes, 3,000 emails/month
- Sign up at https://console.agentmail.to
- The backend reads `AGENTMAIL_API_KEY` from the agent container's `.env` via docker exec
