
"use client"

import { useEffect, useState } from "react"
import { Navbar } from "@/components/Navbar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import api from "@/lib/api"
import { PlayCircle } from "lucide-react"

interface ContentItem {
    video_url: string
    summary: string
    key_points: string[]
    image_file: string
}

export default function EducationPage() {
  const [content, setContent] = useState<Record<string, ContentItem>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchContent() {
        try {
            const res = await api.get("/educational/")
            setContent(res.data)
        } catch (error) {
            console.error("Failed to load educational content")
        } finally {
            setLoading(false)
        }
    }
    fetchContent()
  }, [])

  return (
    <div className="min-h-screen bg-muted/20 pb-12">
      <Navbar />
      <div className="container px-4 py-8">
        <h1 className="text-3xl font-bold mb-2">Educational Hub</h1>
        <p className="text-muted-foreground mb-8">Curated resources to help you understand heart health better.</p>

        {loading ? (
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                 {[1,2,3].map(i => (
                     <div key={i} className="h-64 bg-gray-200 animate-pulse rounded-lg"></div>
                 ))}
            </div>
        ) : (
             <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {Object.entries(content).map(([title, item], idx) => (
                    <Card key={idx} className="flex flex-col h-full hover:shadow-lg transition-shadow">
                        <div className="aspect-video relative bg-slate-100 overflow-hidden rounded-t-lg">
                             {/* Optimized Image placeholder (real app would use next/image with full URL) */}
                             {/* Using a placeholder gradient for now or a generic medical SVG if simpler */}
                             <div className="absolute inset-0 flex items-center justify-center bg-gray-200">
                                <span className="text-gray-400 font-medium">Image: {item.image_file}</span>
                             </div>
                        </div>
                        <CardHeader>
                            <CardTitle className="leading-tight">{title}</CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1">
                            <p className="text-sm text-muted-foreground mb-4 line-clamp-3">
                                {item.summary}
                            </p>
                            <h4 className="text-xs font-bold uppercase tracking-wider text-primary mb-2">Key Points</h4>
                            <ul className="list-disc list-inside text-xs space-y-1 text-muted-foreground">
                                {item.key_points.slice(0, 2).map((point, i) => (
                                    <li key={i} dangerouslySetInnerHTML={{ __html: point.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>') }} />
                                ))}
                            </ul>
                        </CardContent>
                        <CardFooter>
                            <Button variant="secondary" className="w-full" asChild>
                                <a href={item.video_url} target="_blank" rel="noopener noreferrer">
                                    <PlayCircle className="mr-2 h-4 w-4" /> Watch Video
                                </a>
                            </Button>
                        </CardFooter>
                    </Card>
                ))}
            </div>
        )}
      </div>
    </div>
  )
}
