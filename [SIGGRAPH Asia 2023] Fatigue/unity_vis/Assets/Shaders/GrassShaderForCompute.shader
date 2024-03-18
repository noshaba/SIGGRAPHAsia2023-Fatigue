Shader "Custom/GrassForCompute"
{
    Properties
    {
        [Toggle(FADE)] _TransparentBottom("Transparency at Bottom", Float) = 0
        _Fade("Fade Multiplier", Range(1,10)) = 6
    }

    CGINCLUDE
    #include "UnityCG.cginc" 
    #include "Lighting.cginc"
    #include "AutoLight.cginc"
    #pragma multi_compile _SHADOWS_SCREEN
    #pragma multi_compile_fwdbase_fullforwardshadows
    #pragma multi_compile_fog
    #pragma shader_feature FADE

    //
    struct DrawVertex
    {
        float3 positionWS; // The position in world space 
        float2 uv;
        float3 diffuseColor;
    };

    // A triangle on the generated mesh
    struct DrawTriangle
    {
        float3 normalOS;
        DrawVertex vertices[3]; // The three points on the triangle
    };

    StructuredBuffer<DrawTriangle> _DrawTriangles;

    struct v2f
    {
        float4 pos : SV_POSITION; // Position in clip space
        float2 uv : TEXCOORD0;          // The height of this vertex on the grass blade
        float3 positionWS : TEXCOORD1; // Position in world space
        float3 normalWS : TEXCOORD2;   // Normal vector in world space
        float3 diffuseColor : COLOR;
        LIGHTING_COORDS(3, 4)
        UNITY_FOG_COORDS(5)
    };

    // Properties
    float4 _TopTint;
    float4 _BottomTint;
    float _AmbientStrength;
    float _Fade;

    // Vertex function
    struct unityTransferVertexToFragmentSucksHack
    {
        float3 vertex : POSITION;
    };

    // -- retrieve data generated from compute shader
    v2f vert(uint vertexID : SV_VertexID)
    {
        // Initialize the output struct
        v2f output = (v2f)0;

        // Get the vertex from the buffer
        // Since the buffer is structured in triangles, we need to divide the vertexID by three
        // to get the triangle, and then modulo by 3 to get the vertex on the triangle
        DrawTriangle tri = _DrawTriangles[vertexID / 3];
        DrawVertex input = tri.vertices[vertexID % 3];

        output.pos = UnityObjectToClipPos(input.positionWS);
        output.positionWS = input.positionWS;
        
        // float3 faceNormal = GetMainLight().direction * tri.normalOS;
        float3 faceNormal = tri.normalOS;
        // output.normalWS = TransformObjectToWorldNormal(faceNormal, true);
        output.normalWS = faceNormal;
        
        output.uv = input.uv;

        output.diffuseColor = input.diffuseColor;

        // making pointlights work requires v.vertex
        unityTransferVertexToFragmentSucksHack v;
        v.vertex = output.pos;

        TRANSFER_VERTEX_TO_FRAGMENT(output);
        UNITY_TRANSFER_FOG(output,  output.pos);

        return output;
    }


    
    
    ENDCG
    SubShader
    {
        Cull Off
        Blend SrcAlpha OneMinusSrcAlpha // for the transparency
        Pass // basic color with directional lights
        {
            Tags
            {              
                "LightMode" = "ForwardBase"
            }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag            
            
            float4 frag(v2f i) : SV_Target
            {
                
                // take shadow data
                float shadow = 1;
                #if defined(SHADOWS_SCREEN)
                    shadow = (SAMPLE_DEPTH_TEXTURE_PROJ(_ShadowMapTexture, UNITY_PROJ_COORD(i._ShadowCoord)).r);
                #endif			
                // base color by lerping 2 colors over the UVs
                float4 baseColor = lerp(_BottomTint , _TopTint , saturate(i.uv.y)) * float4(i.diffuseColor, 1);
                // multiply with lighting color
                float4 litColor = (baseColor * _LightColor0);
                // multiply with vertex color, and shadows
                float4 final = litColor;
                final.rgb = litColor * shadow;
                // add in baseColor when lights turned off
                final += saturate((1 - shadow) * baseColor * 0.2);
                // add in ambient color
                final += (unity_AmbientSky * baseColor * _AmbientStrength);
                
                // add fog
                UNITY_APPLY_FOG(i.fogCoord, final);
                // fade the bottom based on the vertical uvs
                #if FADE
                    float alpha = lerp(0, 1, saturate(i.uv.y * _Fade));
                    final.a = alpha;
                #endif
                return final;               
            }
            ENDCG
        }

        Pass
        // point lights
        {
            Tags
            {              
                "LightMode" = "ForwardAdd"
            }
            Blend OneMinusDstColor One
            ZWrite Off
            Cull Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag									
            #pragma multi_compile_fwdadd_fullforwardshadows

            float4 frag(v2f i) : SV_Target
            {
                UNITY_LIGHT_ATTENUATION(atten, i, i.positionWS);

                // base color by lerping 2 colors over the UVs
                float3 baseColor = lerp(_BottomTint , _TopTint , saturate(i.uv.y)) * i.diffuseColor;
                
                float3 pointlights = atten * _LightColor0.rgb * baseColor;
                #if FADE
                    float alpha = lerp(0, 1, saturate(i.uv.y * _Fade));
                    pointlights *= alpha;
                #endif

                return float4(pointlights, 1);
            }
            ENDCG
        }

        Pass // shadow pass
        {
            
            Tags
            {
                "LightMode" = "ShadowCaster"
            }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_shadowcaster

            float4 frag(v2f i) : SV_Target
            {

                SHADOW_CASTER_FRAGMENT(i)
            }
            ENDCG
        }


    }  	  Fallback "VertexLit"
}
