
"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "./ui/button"
import { HeartPulse } from "lucide-react"

export function Navbar() {
  const pathname = usePathname()

  const routes = [
    { href: "/", label: "Home" },
    { href: "/predict", label: "Single Prediction" },
    { href: "/bulk", label: "Bulk Analysis" },
    { href: "/resources", label: "Resources" },
    { href: "/education", label: "Education" },
  ]

  return (
    <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="container flex h-16 items-center px-4">
        <Link href="/" className="mr-8 flex items-center space-x-2">
          <HeartPulse className="h-6 w-6 text-primary" />
          <span className="font-bold sm:inline-block">CardioHealth</span>
        </Link>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
             {/* Mobile Nav could go here */}
          </div>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            {routes.map((route) => (
              <Link
                key={route.href}
                href={route.href}
                className={cn(
                  "transition-colors hover:text-foreground/80",
                  pathname === route.href ? "text-foreground" : "text-foreground/60"
                )}
              >
                {route.label}
              </Link>
            ))}
            <Link href="/login">
                <Button variant="outline" size="sm">Login</Button>
            </Link>
          </nav>
        </div>
      </div>
    </nav>
  )
}
