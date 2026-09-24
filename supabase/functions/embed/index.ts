import "jsr:@supabase/functions-js/edge-runtime.d.ts"

// Initialize the native gte-small embedding model session once in edge runtime
const session = new Supabase.ai.Session('gte-small')

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

Deno.serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })
  }

  try {
    const body = await req.json()
    const input = body?.input

    if (!input || (Array.isArray(input) && input.length === 0)) {
      return new Response(
        JSON.stringify({ error: 'Missing or empty required "input" field' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )
    }

    // Support both single string and array of strings
    const texts: string[] = Array.isArray(input) ? input : [input]
    const embeddings: number[][] = []

    for (const text of texts) {
      if (typeof text !== 'string') {
        return new Response(
          JSON.stringify({ error: 'All input items must be strings' }),
          {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        )
      }

      // Generate 384-dimensional normalized embedding vector via Supabase.ai
      const output = await session.run(text, {
        mean_pool: true,
        normalize: true,
      })

      // Convert Float32Array to standard JavaScript array of numbers
      embeddings.push(Array.from(output))
    }

    return new Response(
      JSON.stringify({ embeddings }),
      {
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  } catch (err: any) {
    return new Response(
      JSON.stringify({ error: err.message || 'Internal server error during embedding generation' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})
