exports.handler = async (event, context) => {
  // CORS headers
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS'
  };

  // Handle preflight CORS requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  // Extract the API path from the Netlify function path
  const apiPath = event.path.replace('/.netlify/functions/proxy', '') || '';
  const queryString = event.queryStringParameters ? 
    '?' + new URLSearchParams(event.queryStringParameters).toString() : '';
  const targetUrl = `http://13.48.123.148:8000${apiPath}${queryString}`;

  console.log('=== PROXY DEBUG ===');
  console.log('Original path:', event.path);
  console.log('Extracted API path:', apiPath);
  console.log('Target URL:', targetUrl);
  console.log('Method:', event.httpMethod);
  console.log('Headers:', event.headers);

  try {
    // Use built-in fetch (available in Node.js 18+)
    // No import needed as fetch is globally available

    const options = {
      method: event.httpMethod,
      headers: {}
    };

    // Handle different content types
    if (event.httpMethod === 'POST' && event.body) {
      if (event.isBase64Encoded) {
        // Handle binary data (like file uploads)
        options.body = Buffer.from(event.body, 'base64');
      } else {
        options.body = event.body;
      }

      // Copy content-type header if present
      if (event.headers['content-type']) {
        options.headers['Content-Type'] = event.headers['content-type'];
      }
    }

    const response = await fetch(targetUrl, options);
    
    // Get response data
    const contentType = response.headers.get('content-type') || '';
    let responseBody;

    if (contentType.includes('application/json')) {
      responseBody = JSON.stringify(await response.json());
    } else if (contentType.includes('image/')) {
      // Handle image responses
      const buffer = await response.buffer();
      responseBody = buffer.toString('base64');
      return {
        statusCode: response.status,
        headers: {
          ...headers,
          'Content-Type': contentType
        },
        body: responseBody,
        isBase64Encoded: true
      };
    } else {
      responseBody = await response.text();
    }

    return {
      statusCode: response.status,
      headers: {
        ...headers,
        'Content-Type': contentType
      },
      body: responseBody
    };

  } catch (error) {
    console.error('Proxy error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({
        error: 'Proxy request failed',
        details: error.message,
        targetUrl: targetUrl
      })
    };
  }
};
