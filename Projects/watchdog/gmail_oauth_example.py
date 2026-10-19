from __future__ import print_function
import os
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If you only need read-only access, use this scope.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print('Error: credentials.json not found. Follow README instructions to create it.')
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])
    if not messages:
        print('No messages found.')
        return
    print('Message IDs:')
    for m in messages:
        print(m['id'])
    # Fetch first message snippet
    msg = service.users().messages().get(userId='me', id=messages[0]['id'], format='full').execute()
    print('\nSnippet:', msg.get('snippet'))

if __name__ == '__main__':
    main()
