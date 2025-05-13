FROM node:22-alpine AS build

WORKDIR /app
# Set environment
ENV NODE_ENV=production
# Copy package files
COPY src/frontend/package*.json ./

# Install dependencies: Runs a clean install of dependencies based on package-lock.json file
RUN npm ci

# Copy all frontend files
COPY src/frontend/ ./

# Create public directory if it doesn't exist
RUN mkdir -p public

# Executes the build script defined in package.json file, which compiles the Next.js application for production.
RUN npm run build

# Production stage
# Starts a new stage named "runner" using the same base image. Multi-stage builds help create smaller production images.
FROM node:22-alpine AS runner
# Set environment
ENV NODE_ENV=production

# Set working directory
WORKDIR /app

# Copy package files
COPY --from=build /app/package*.json ./

# Install only production dependencies
RUN npm ci --only=production

# Copy built application
# .next directory contains the compiled Next.js application - all JS, CSS, and other assets that were processed during the build step
COPY --from=build /app/.next ./.next
# public folder contains static files like images, fonts, or other assets that Next.js serves directly
COPY --from=build /app/public ./public
# next.config.js file contains runtime configuration that Next.js needs when running in production
COPY --from=build /app/next.config.js ./
# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs
# Switch to non-root user
USER nextjs
# Expose port
EXPOSE 3000

# Start Next.js in production mode
CMD ["npm", "start"]