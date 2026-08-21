# Static build of the overlay viewer, served by nginx.

FROM node:22-alpine AS build

WORKDIR /app

# Lockfile-first for layer caching. `npm ci` rather than `npm install` so the image is built
# from the exact resolved tree the tests ran against, not from whatever satisfies the ranges
# on the day of the build.
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Baked at build time because Vite inlines env vars into the bundle -- there is no runtime
# environment in a static build to read them from. Compose passes the host-visible address of
# the API, which is what the *browser* must reach; the compose-internal name would not resolve
# in the user's browser.
ARG VITE_API_BASE_URL=http://localhost:8080
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=4s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1
