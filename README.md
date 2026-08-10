# Vigilant Eye

PROJECT OVERVIEW

This project is a graduation prototype for a generic AI-powered IP camera monitoring and intelligent event detection system.

The web application must support multiple IP cameras in its architecture, although the graduation demonstration may use only one physically connected camera.

The implemented AI use case is mobile phone detection during examinations. When a person is detected using or holding a mobile phone for a configured duration, the system creates a “Suspicious Cheating Activity” event.

The system must never describe AI detections as confirmed cheating. It must use terms such as:

- Suspicious Cheating Activity

- Possible Cheating Activity

- Mobile Phone Detected

SYSTEM ARCHITECTURE

The complete system consists of:

1. IP cameras and/or NVR.

2. A separate Python AI service running on a Windows Server or another local machine.

3. A Lovable web application.

4. Supabase for authentication, database, storage and real-time event updates.

5. REST API and WebSocket communication between the Python AI service and the web application.

The browser application must not contain the computer vision model itself.

The Python AI service will:

- Receive RTSP camera streams.

- Run YOLO object detection.

- Detect people and mobile phones.

- draw detection boxes.

- Generate alerts.

- Save snapshots.

- Send events and system status to the web application through an API.

The Lovable application will:

- Handle login and logout.

- Display the dashboard.

- Manage cameras.

- Display live streams.

- Display AI events and alerts.

- Manage AI rules and settings.

- Store and retrieve event records.

- Manage users and roles.

DESIGN REQUIREMENTS

The visual design must closely follow the uploaded reference screenshots.

Design style:

- Generic AI surveillance command-center interface.

- Deep navy and charcoal background.

- Cyan, teal and green system-status accents.

- Red for critical alerts.

- Amber/yellow for warnings.

- Thin glowing borders.

- Compact technical typography.

- Subtle grid and HUD-style details.

- Professional, realistic and functional.

- Avoid a generic SaaS dashboard appearance.

- Avoid excessive gradients.

- Avoid oversized cards.

- Avoid cartoon illustrations.

- Avoid excessive rounded corners.

- Use subtle animations only.

The interface must remain clear and readable on a 1920×1080 display during the graduation presentation.

APPLICATION PAGES

Required routes:

- /login

- /dashboard

- /monitoring

- /events

- /cameras

- /ai-rules

- /reports

- /users

- /settings

- /profile

AUTHENTICATION AND ROLES

Use secure email and password authentication.

Roles:

- Administrator

- Operator

Administrator permissions:

- Manage cameras.

- Manage users.

- Configure AI rules.

- Access all events and reports.

- Change system settings.

Operator permissions:

- View cameras and streams.

- Receive alerts.

- Review events.

- Confirm or reject AI events.

- Cannot manage users or critical system settings.

All protected routes must redirect unauthenticated users to /login.

MAIN AI USE CASE

The initial functional detection rule is:

Mobile Phone Cheating Detection

Suggested rule settings:

- Confidence threshold.

- Minimum detection duration.

- Alert cooldown.

- Severity level.

- Enabled cameras.

- Snapshot saving.

- Sound notification.

Other future detection rules may appear as disabled or “Coming Soon”, but they must not be presented as currently functional.

GENERIC MULTI-CAMERA REQUIREMENT

Do not hard-code the application for one camera.

The interface must calculate dynamically:

- Total configured cameras.

- Online cameras.

- Offline cameras.

- Cameras with AI enabled.

- Cameras currently recording.

The initial demo database may contain one real camera and optional clearly labeled demonstration cameras.

DATA RULES

Never hard-code dashboard statistics in final production components.

Use service layers and reusable hooks for:

- Cameras.

- Events.

- AI service status.

- NVR status.

- Users.

- Reports.

During the UI stage, use structured mock data through a replaceable mock service.

Later, the mock service will be replaced with Supabase and the Python AI REST API.

TECHNICAL RULES

- Use React and TypeScript.

- Use Tailwind CSS.

- Prefer reusable shadcn/ui components where appropriate.

- Use responsive layouts.

- Use clean reusable components.

- Do not put all logic in one file.

- Use typed interfaces.

- Do not use TypeScript any.

- Separate UI components, services, hooks and data types.

- Do not expose camera passwords or secret keys in frontend code.

- Do not put RTSP credentials directly in visible UI output.

- Do not add payment, signup, social login or unrelated SaaS functionality.

- Do not redesign existing completed pages unless explicitly requested.

- Preserve the reference visual identity throughout all future pages.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/6955f79f-448a-4f4c-b1c1-75ac4b9cbbc1).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
