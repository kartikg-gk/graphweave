import { useClerk, useSignIn, useSignUp } from "@clerk/clerk-react";
import { isClerkAPIResponseError } from "@clerk/clerk-react/errors";
import { useState, type FormEvent, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";

type Mode = "sign-in" | "sign-up";
type OAuth = "oauth_github" | "oauth_google";

const inputClass =
  "w-full rounded-sm border border-line bg-paper px-3 py-2 text-sm text-ink outline-none focus:border-blue";

/** The provider's message for a failed call, or a plain fallback. */
function messageOf(error: unknown): string {
  if (isClerkAPIResponseError(error)) {
    const first = error.errors[0];
    return first?.longMessage || first?.message || "That didn't work. Try again.";
  }
  return "Something went wrong. Try again.";
}

function Field({ id, label, children }: { id: string; label: string; children: ReactNode }) {
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-xs text-ink-dim">{label}</label>
      {children}
    </div>
  );
}

/**
 * Sign-in and sign-up drawn by the app rather than by the provider's
 * prebuilt components, so the page carries the product's look and name only.
 *
 * Email with password, a one-time email code instead of a password, and
 * GitHub or Google. Whatever the account needs next (a code sent to the
 * address) is asked for on the same card.
 */
export function AuthForm({ mode }: { mode: Mode }) {
  const signInState = useSignIn();
  const signUpState = useSignUp();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  // What the card is waiting for: the first form, or a code sent by email.
  const [step, setStep] = useState<"start" | "code">("start");
  const [codeFor, setCodeFor] = useState<"first" | "second" | "sign-up">("first");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = signInState.isLoaded && signUpState.isLoaded;

  async function finish(sessionId: string | null) {
    if (!sessionId) return;
    const setActive = mode === "sign-in" ? signInState.setActive : signUpState.setActive;
    await setActive?.({ session: sessionId });
    navigate("/", { replace: true });
  }

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (cause) {
      setError(messageOf(cause));
    } finally {
      setBusy(false);
    }
  }

  function oauth(strategy: OAuth) {
    void run(async () => {
      const target = mode === "sign-in" ? signInState.signIn : signUpState.signUp;
      await target?.authenticateWithRedirect({
        strategy,
        redirectUrl: "/sso-callback",
        redirectUrlComplete: "/",
      });
    });
  }

  async function sendSignInCode() {
    const signIn = signInState.signIn!;
    const attempt = await signIn.create({ identifier: email });
    const factor = attempt.supportedFirstFactors?.find((item) => item.strategy === "email_code");
    if (!factor || !("emailAddressId" in factor)) throw new Error("Email codes are not available for this account.");
    await signIn.prepareFirstFactor({ strategy: "email_code", emailAddressId: factor.emailAddressId });
    setCodeFor("first");
    setStep("code");
  }

  function submitStart(event: FormEvent) {
    event.preventDefault();
    void run(async () => {
      if (mode === "sign-in") {
        const attempt = await signInState.signIn!.create({ identifier: email, password });
        if (attempt.status === "complete") return finish(attempt.createdSessionId);
        if (attempt.status === "needs_second_factor") {
          // A new device can be asked to confirm with a code sent by email.
          await signInState.signIn!.prepareSecondFactor({ strategy: "email_code" });
          setCodeFor("second");
          setStep("code");
          return;
        }
        throw new Error(`Sign-in needs a step this form doesn't support (${attempt.status}).`);
      }
      await signUpState.signUp!.create({ emailAddress: email, password });
      await signUpState.signUp!.prepareEmailAddressVerification({ strategy: "email_code" });
      setCodeFor("sign-up");
      setStep("code");
    });
  }

  function submitCode(event: FormEvent) {
    event.preventDefault();
    void run(async () => {
      if (codeFor === "sign-up") {
        const attempt = await signUpState.signUp!.attemptEmailAddressVerification({ code });
        if (attempt.status === "complete") return finish(attempt.createdSessionId);
        throw new Error("Your account needs more details than this form collects.");
      }
      const signIn = signInState.signIn!;
      const attempt = codeFor === "first"
        ? await signIn.attemptFirstFactor({ strategy: "email_code", code })
        : await signIn.attemptSecondFactor({ strategy: "email_code", code });
      if (attempt.status === "complete") return finish(attempt.createdSessionId);
      throw new Error("That code didn't finish signing you in. Try again.");
    });
  }

  const title = mode === "sign-in" ? "Sign in to graphweave" : "Create your graphweave account";

  return (
    <div className="w-full max-w-sm rounded-md border border-line bg-panel p-6 shadow-sm">
      <h1 className="text-lg font-semibold text-ink">{title}</h1>
      <p className="mt-1 text-sm text-ink-dim">
        {step === "code" ? `Enter the code sent to ${email}.` : "Your questions and answers are saved to your account."}
      </p>

      {error && <p role="alert" className="mt-4 rounded-sm border border-red/40 bg-red/10 p-2 text-sm text-red">{error}</p>}

      {step === "start" ? (
        <>
          <div className="mt-5 grid gap-2">
            <button type="button" className="page-action justify-center" disabled={!ready || busy} onClick={() => oauth("oauth_github")}>Continue with GitHub</button>
            <button type="button" className="page-action justify-center" disabled={!ready || busy} onClick={() => oauth("oauth_google")}>Continue with Google</button>
          </div>

          <div className="my-5 flex items-center gap-3 text-xs text-ink-muted">
            <span className="h-px flex-1 bg-line" />or<span className="h-px flex-1 bg-line" />
          </div>

          <form className="space-y-3" onSubmit={submitStart}>
            <Field id="auth-email" label="Email">
              <input id="auth-email" className={inputClass} type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} />
            </Field>
            <Field id="auth-password" label="Password">
              <input id="auth-password" className={inputClass} type="password" required minLength={8}
                autoComplete={mode === "sign-in" ? "current-password" : "new-password"}
                value={password} onChange={(event) => setPassword(event.target.value)} />
            </Field>
            {/* The provider's bot check renders here when sign-up needs one. */}
            {mode === "sign-up" && <div id="clerk-captcha" />}
            <button type="submit" className="page-action w-full justify-center" disabled={!ready || busy}>
              {busy ? "Working…" : mode === "sign-in" ? "Sign in" : "Create account"}
            </button>
          </form>

          {mode === "sign-in" && (
            <button type="button" className="mt-3 text-xs text-ink-dim underline underline-offset-4 disabled:opacity-50"
              disabled={!ready || busy || !email}
              onClick={() => void run(sendSignInCode)}>
              Email me a code instead
            </button>
          )}
        </>
      ) : (
        <form className="mt-5 space-y-3" onSubmit={submitCode}>
          <Field id="auth-code" label="Code">
            <input id="auth-code" className={inputClass} inputMode="numeric" autoComplete="one-time-code" required value={code} onChange={(event) => setCode(event.target.value.trim())} />
          </Field>
          <button type="submit" className="page-action w-full justify-center" disabled={busy}>{busy ? "Checking…" : "Continue"}</button>
          <button type="button" className="text-xs text-ink-dim underline underline-offset-4" onClick={() => { setStep("start"); setCode(""); setError(null); }}>Start over</button>
        </form>
      )}

      <p className="mt-6 text-sm text-ink-dim">
        {mode === "sign-in"
          ? <>New here? <Link className="text-ink underline underline-offset-4" to="/sign-up">Create an account</Link></>
          : <>Have an account? <Link className="text-ink underline underline-offset-4" to="/sign-in">Sign in</Link></>}
      </p>
    </div>
  );
}

/** Signed-in control for the top bar: the address and a sign-out button. */
export function AccountControl({ email }: { email: string | null }) {
  const { signOut } = useClerk();
  return (
    <span className="flex items-center gap-2">
      {email && <span className="hidden max-w-[16ch] truncate text-xs text-ink-dim sm:inline" title={email}>{email}</span>}
      <button type="button" className="page-action whitespace-nowrap" onClick={() => void signOut({ redirectUrl: "/" })}>Sign out</button>
    </span>
  );
}
