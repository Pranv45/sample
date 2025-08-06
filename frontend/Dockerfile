# Use Node.js 18 Alpine as base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy ALL application files first (respecting .dockerignore)
COPY . .

# Install dependencies based on the copied package.json
RUN npm install

# Build the app for production
RUN npm run build

# Install serve to run the app
RUN npm install -g serve

# Expose port 3000
EXPOSE 3000

# Start the app
CMD ["serve", "-s", "build", "-l", "3000"]