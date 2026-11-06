export class AuthError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'AuthError';
    this.status = status;
  }
}

const messages = new Map([
  [400, 'Check the username and password format, then try again.'],
  [401, 'That username or password did not match.'],
  [409, 'That username is already in use. Try signing in instead.'],
]);

async function request(path, { method = 'GET', body, fetchImpl = globalThis.fetch?.bind(globalThis) } = {}) {
  const options = { method, credentials: 'same-origin', headers: { Accept: 'application/json' } };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetchImpl(path, options);
  } catch {
    throw new AuthError('Could not reach Personalized Memory sign-in. Check your connection and try again.', 0);
  }
  if (!response.ok) {
    throw new AuthError(messages.get(response.status) ?? 'Sign-in could not be completed. Please try again.', response.status);
  }
  if (response.status === 204) return null;
  try {
    return await response.json();
  } catch {
    throw new AuthError('Personalized Memory returned an unreadable sign-in response.', response.status);
  }
}

export async function getAuthSession(fetchImpl) {
  const result = await request('/api/auth/session', { fetchImpl });
  if (!result || typeof result.authenticated !== 'boolean') {
    throw new AuthError('Personalized Memory returned an invalid sign-in session.', 200);
  }
  return {
    authenticated: result.authenticated,
    username: result.authenticated && typeof result.username === 'string' ? result.username : '',
  };
}

export async function login(username, password, fetchImpl) {
  return request('/api/auth/login', { method: 'POST', body: { username, password }, fetchImpl });
}

export async function register(username, password, fetchImpl) {
  return request('/api/auth/register', { method: 'POST', body: { username, password }, fetchImpl });
}

export async function logout(fetchImpl) {
  return request('/api/auth/logout', { method: 'POST', fetchImpl });
}
