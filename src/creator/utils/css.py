# Add this near the top of your app.py file
css = """
:root {
  --main-text-size: 16px;
  --heading-text-size: 24px;
  --subheading-text-size: 20px;
  --button-text-size: 16px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  --primary-color: #2563EB;
  --secondary-color: #4B5563;
  --success-color: #10B981;
  --warning-color: #F59E0B;
  --error-color: #EF4444;
}

body, .gradio-container {
  font-size: var(--main-text-size) !important;
  line-height: 1.5;
}

h1 {
  font-size: 32px !important;
  font-weight: 700 !important;
  margin-bottom: var(--spacing-lg) !important;
  color: var(--primary-color);
}

h2 {
  font-size: 28px !important;
  font-weight: 600 !important;
  margin-bottom: var(--spacing-md) !important;
}

h3 {
  font-size: var(--subheading-text-size) !important;
  font-weight: 600 !important;
  margin-bottom: var(--spacing-md) !important;
}

/* Buttons */
button.primary {
  background-color: var(--primary-color) !important;
  color: white !important;
  font-size: var(--button-text-size) !important;
  padding: 10px 20px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}

button.secondary {
  background-color: var(--secondary-color) !important;
  color: white !important;
  font-size: var(--button-text-size) !important;
  padding: 8px 16px !important;
  border-radius: 8px !important;
}

button.success {
  background-color: var(--success-color) !important;
  color: white !important;
  font-size: var(--button-text-size) !important;
  padding: 10px 20px !important;
  border-radius: 8px !important;
}

/* Section spacing */
.section {
  margin-bottom: var(--spacing-lg) !important;
  padding: var(--spacing-md) !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
  background-color: white !important;
}

/* Status messages */
.status-box {
  font-size: var(--main-text-size) !important;
  padding: 10px !important;
  border-radius: 8px !important;
  margin-top: 8px !important;
}

/* Improved chatbot appearance */
.chatbot-container {
  height: 500px !important;
  border-radius: 12px !important;
}

.conversation-container {
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
  padding: 16px !important;
  margin-bottom: 16px !important;
  background-color: #f9fafb !important;
}

/* Improved spacing between elements */
.gradio-row {
  margin-bottom: var(--spacing-md) !important;
}

/* Better dropdown appearance */
select, .gradio-dropdown {
  font-size: var(--main-text-size) !important;
  padding: 8px 12px !important;
  border-radius: 6px !important;
}

/* Text inputs */
textarea, input[type="text"] {
  font-size: var(--main-text-size) !important;
  padding: 10px 12px !important;
  border-radius: 6px !important;
  border: 1px solid #d1d5db !important;
}
"""